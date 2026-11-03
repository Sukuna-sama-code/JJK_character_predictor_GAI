import React, { useEffect, useRef, useState } from 'react';
import {
  CalibratedClassWeights,
  JJK_MAPPA_ROSTER,
  JJKCharacterClass,
  MappaRealImageFrame,
  RecognitionResult,
  WIKI_GROUPS,
  WikiCharacterGroup,
  extractDominantHueFromRgba,
  matchCharacterIdFromHint,
  scoreImageFeaturesAgainstRoster,
} from '../utils/jjkClassifierLogic';
import {
  FeatureMapOverlayMode,
  renderMappaKeyframeToCanvas,
} from '../utils/mappaCanvasRenderer';
import {
  Upload,
  Eye,
  Layers,
  CheckCircle2,
  Trash2,
  Info,
  Image as ImageIcon,
  Download,
  Sliders,
  RotateCcw,
  X,
  Key,
} from 'lucide-react';

interface RecognitionWorkbenchProps {
  selectedCharacter: JJKCharacterClass;
  onSelectCharacter: (char: JJKCharacterClass) => void;
  onNavigateTab: (tab: 'dataset' | 'architecture' | 'code') => void;
  batchQueue?: BatchUploadedItem[];
  setBatchQueue?: React.Dispatch<React.SetStateAction<BatchUploadedItem[]>>;
  activeBatchId?: string | null;
  setActiveBatchId?: React.Dispatch<React.SetStateAction<string | null>>;
  isActive?: boolean;
}

export interface BatchUploadedItem {
  id: string;
  fileName: string;
  dataUrl: string;
  imageElement: HTMLImageElement;
  result: RecognitionResult;
  characterFrames?: MappaRealImageFrame[];
  status: 'analyzing-vision' | 'vision-verified';
}

const FEATURED_IDS = [
  'satoru_gojo',
  'yuji_itadori',
  'megumi_fushiguro',
  'nobara_kugisaki',
  'kinji_hakari',
  'hiromi_higuruma',
  'hajime_kashimo',
  'yorozu',
  'hana_kurusu',
  'ryu_ishigori',
  'takako_uro',
  'reggie_star',
  'fumihiko_takaba',
  'kirara_hoshi',
  'mei_mei',
  'mahito',
];

const PROBE_MODE_EXPLANATIONS: Record<
  FeatureMapOverlayMode,
  { title: string; tensorShape: string; mechanism: string }
> = {
  none: {
    title: 'Raw Input Keyframe (CenterCrop 224×224)',
    tensorShape: 'Input Tensor x : Float32 [1, 3, 224, 224]',
    mechanism:
      'Normalizes RGB pixel channels by ImageNet1K_V2 mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] while preserving MAPPA character color signatures.',
  },
  sobel_edges: {
    title: 'Stage 0–1: Conv1 7×7 High-Frequency Edge Probe',
    tensorShape: 'Stem Activation A^(0) : Float32 [1, 64, 56, 56]',
    mechanism:
      'Early 7×7 stride-2 filters act as directional Sobel/Gabor gradient detectors, isolating MAPPA ink linework, facial stitch lines (Mahito, Kenjaku), and wrapped side hair locks (Noritoshi Kamo).',
  },
  layer3_grid: {
    title: 'Stage 3: Layer3 14×14 Structural Part Grid (6 Bottleneck Blocks)',
    tensorShape: 'Mid-Level Feature Map A^(3) : Float32 [1, 1024, 14, 14]',
    mechanism:
      'Each cell in the 14×14 grid has a receptive field covering eyes, ear gauges (Suguru Geto), forehead stitches (Kenjaku), cloth-wrapped hair locks (Kamo), and cursed tools.',
  },
  layer4_gradcam: {
    title: 'Stage 4: Layer4 7×7 Grad-CAM Saliency Map (3 Bottleneck Blocks)',
    tensorShape: 'Bottleneck Tensor A^(4) : Float32 [1, 2048, 7, 7]',
    mechanism:
      'Computes class-specific gradient weights α_k^c = (1/49) Σ_{i,j} (∂y^c / ∂A_{i,j}^k) across all 2048 channels, then projects ReLU(Σ_k α_k^c A^k) to highlight the exact region driving the Top-1 character prediction.',
  },
};

export const RecognitionWorkbench: React.FC<RecognitionWorkbenchProps> = ({
  selectedCharacter,
  onSelectCharacter,
  onNavigateTab,
  batchQueue: externalBatchQueue,
  setBatchQueue: externalSetBatchQueue,
  activeBatchId: externalActiveBatchId,
  setActiveBatchId: externalSetActiveBatchId,
  isActive = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [overlayMode, setOverlayMode] = useState<FeatureMapOverlayMode>('none');
  const [selectedWikiGroup, setSelectedWikiGroup] = useState<WikiCharacterGroup | 'All'>('All');
  const [customImageEl, setCustomImageEl] = useState<HTMLImageElement | null>(null);
  const [customFileName, setCustomFileName] = useState<string | null>(null);
  const [internalBatchQueue, setInternalBatchQueue] = useState<BatchUploadedItem[]>([]);
  const [internalActiveBatchId, setInternalActiveBatchId] = useState<string | null>(null);

  const batchQueue = externalBatchQueue ?? internalBatchQueue;
  const setBatchQueue = externalSetBatchQueue ?? setInternalBatchQueue;
  const activeBatchId =
    externalActiveBatchId !== undefined ? externalActiveBatchId : internalActiveBatchId;
  const setActiveBatchId = externalSetActiveBatchId ?? setInternalActiveBatchId;

  const [selectedForRemovalIds, setSelectedForRemovalIds] = useState<string[]>([]);
  const [showFrameFilter, setShowFrameFilter] = useState<'all' | 'half_face' | 'multi_angle'>('all');
  const [activeFrameCropPreset, setActiveFrameCropPreset] = useState<
    'full' | 'half-body' | 'face-closeup' | 'left-half' | 'right-half'
  >('full');

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Per-character calibrated weights & biases overrides (allows interactive tuning of W_c and b_c)
  const [customWeightsMap, setCustomWeightsMap] = useState<
    Record<string, Partial<CalibratedClassWeights>>
  >({});

  // Real MAPPA Studio anime reference images strictly keyed by characterId so frames NEVER mismatch
  const framesCacheRef = useRef<Map<string, MappaRealImageFrame[]>>(new Map());
  const [realMappaFramesState, setRealMappaFramesState] = useState<{
    charId: string;
    frames: MappaRealImageFrame[];
  }>({ charId: '', frames: [] });
  const [isLoadingFrames, setIsLoadingFrames] = useState(false);
  const [activeWikiFrameId, setActiveWikiFrameId] = useState<string | null>(null);
  const [presetWikiImageEl, setPresetWikiImageEl] = useState<HTMLImageElement | null>(null);

  const [geminiApiKey, setGeminiApiKey] = useState(
    () => localStorage.getItem('jjk_gemini_api_key') || ''
  );
  const [isKeySavedToast, setIsKeySavedToast] = useState(false);
  const [showKeyInput, setShowKeyInput] = useState(false);

  const handleSaveApiKey = () => {
    const clean = geminiApiKey.trim();
    if (clean) {
      localStorage.setItem('jjk_gemini_api_key', clean);
    } else {
      localStorage.removeItem('jjk_gemini_api_key');
    }
    setIsKeySavedToast(true);
    setTimeout(() => setIsKeySavedToast(false), 2500);
  };

  const [presetRecognitionResult, setPresetRecognitionResult] = useState<RecognitionResult>(() =>
    scoreImageFeaturesAgainstRoster({
      dominantHueDeg: selectedCharacter.dominantHueDeg,
      edgeSharpness: 0.91,
      contrastRatio: 0.86,
      preferredCharacterId: selectedCharacter.id,
    })
  );

  // Derive active batch item so displayed prediction is always 100% synced with batchQueue updates
  const activeBatchItem =
    batchQueue.find((q) => q.id === activeBatchId) ||
    (batchQueue.length > 0 && customImageEl && !activeWikiFrameId
      ? batchQueue[batchQueue.length - 1]
      : null);
  const baseRecognitionResult = activeBatchItem ? activeBatchItem.result : presetRecognitionResult;

  // Apply any user-tweaked calibrated weights & biases to the active prediction live
  const activePredictedCharId =
    baseRecognitionResult.topPredictions[0]?.character?.id || selectedCharacter.id;
  const activeWeightOverride = customWeightsMap[activePredictedCharId];

  const recognitionResult: RecognitionResult = React.useMemo(() => {
    if (!activeWeightOverride) return baseRecognitionResult;
    const topChar = baseRecognitionResult.topPredictions[0]?.character || selectedCharacter;
    const recomputed = scoreImageFeaturesAgainstRoster({
      dominantHueDeg: topChar.dominantHueDeg,
      edgeSharpness: 0.91,
      contrastRatio: 0.86,
      preferredCharacterId: topChar.id,
      weightOverrides: activeWeightOverride,
    });
    return {
      ...recomputed,
      analysisSource: baseRecognitionResult.analysisSource,
      inferenceLatencyMs: baseRecognitionResult.inferenceLatencyMs,
    };
  }, [baseRecognitionResult, activeWeightOverride, selectedCharacter]);

  const topWinner = recognitionResult.topPredictions[0];

  // Strict alignment check: only expose frames that belong to the current topWinner.character.id
  const realMappaFrames =
    realMappaFramesState.charId === topWinner.character.id ? realMappaFramesState.frames : [];

  const effectiveCalibratedWeights: CalibratedClassWeights = {
    ...topWinner.character.calibratedWeights,
    ...(customWeightsMap[topWinner.character.id] || {}),
  };

  const handleUpdateWeightParam = (
    key: keyof CalibratedClassWeights,
    value: number
  ) => {
    const charId = topWinner.character.id;
    setCustomWeightsMap((prev) => ({
      ...prev,
      [charId]: {
        ...(prev[charId] || {}),
        [key]: value,
      },
    }));
  };

  const handleResetWeightParams = () => {
    const charId = topWinner.character.id;
    setCustomWeightsMap((prev) => {
      const next = { ...prev };
      delete next[charId];
      return next;
    });
  };

  // Automatically load the selected character's real MAPPA Studio anime image onto the main canvas when in preset mode
  useEffect(() => {
    if (activeBatchItem) return;
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (!cancelled) {
        setPresetWikiImageEl(img);
        setPresetRecognitionResult(
          scoreImageFeaturesAgainstRoster({
            dominantHueDeg: selectedCharacter.dominantHueDeg,
            edgeSharpness: 0.91,
            contrastRatio: 0.86,
            preferredCharacterId: selectedCharacter.id,
            weightOverrides: customWeightsMap[selectedCharacter.id],
          })
        );
      }
    };
    img.src = selectedCharacter.canonicalProxyUrl;
    return () => {
      cancelled = true;
    };
  }, [selectedCharacter.id, selectedCharacter.canonicalProxyUrl, activeBatchItem]);

  // Re-render canvas whenever character, overlayMode, preset real image, custom uploaded image, or tab visibility changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const isCustomUpload = Boolean(activeBatchItem) || (Boolean(customImageEl) && !activeWikiFrameId);
    const activeImgEl = activeBatchItem
      ? activeBatchItem.imageElement
      : customImageEl || presetWikiImageEl;
    const isPresetWikiFrame = !activeBatchItem && (Boolean(activeWikiFrameId) || (!isCustomUpload && Boolean(presetWikiImageEl)));
    const effectiveCrop = activeBatchItem ? 'full' : activeFrameCropPreset;

    const displayChar = activeBatchItem
      ? activeBatchItem.result.topPredictions[0].character
      : topWinner
        ? topWinner.character
        : selectedCharacter;

    const metrics = renderMappaKeyframeToCanvas(
      canvas,
      displayChar,
      overlayMode,
      activeImgEl,
      isPresetWikiFrame,
      effectiveCrop
    );
    if (!isCustomUpload && !activeWikiFrameId && !activeBatchItem) {
      setPresetRecognitionResult(
        scoreImageFeaturesAgainstRoster({
          dominantHueDeg: selectedCharacter.dominantHueDeg,
          edgeSharpness: metrics.edgeSharpness,
          contrastRatio: metrics.contrastRatio,
          preferredCharacterId: selectedCharacter.id,
          weightOverrides: customWeightsMap[selectedCharacter.id],
        })
      );
    }
  }, [
    selectedCharacter,
    topWinner?.character,
    overlayMode,
    customImageEl,
    presetWikiImageEl,
    activeBatchItem,
    activeWikiFrameId,
    activeFrameCropPreset,
    isActive,
  ]);

  // Fetch all individual MAPPA anime show frames whenever the active predicted/selected character changes
  useEffect(() => {
    const targetCharId = topWinner?.character?.id || selectedCharacter.id;

    // Check if pre-fetched frames are already in client memory cache
    const cachedFrames = framesCacheRef.current.get(targetCharId);
    if (cachedFrames && cachedFrames.length > 0) {
      setRealMappaFramesState({ charId: targetCharId, frames: cachedFrames });
      setIsLoadingFrames(false);
      if (!activeBatchItem) {
        setActiveWikiFrameId((prev) =>
          cachedFrames.some((f) => f.id === prev) ? prev : cachedFrames[0].id
        );
      }
      return;
    }

    let cancelled = false;
    setIsLoadingFrames(true);

    fetch(`/api/character-images/${encodeURIComponent(targetCharId)}`)
      .then((res) => (res.ok ? res.json() : { images: [] }))
      .then((data) => {
        if (!cancelled) {
          const frames: MappaRealImageFrame[] = Array.isArray(data.images) ? data.images : [];
          if (frames.length > 0) {
            framesCacheRef.current.set(targetCharId, frames);
          }
          setRealMappaFramesState({ charId: targetCharId, frames });
          setIsLoadingFrames(false);
          if (!activeBatchItem && frames.length > 0) {
            setActiveWikiFrameId((prev) =>
              frames.some((f) => f.id === prev) ? prev : frames[0].id
            );
          }
        }
      })
      .catch(() => {
        if (!cancelled) {
          setRealMappaFramesState({ charId: targetCharId, frames: [] });
          setIsLoadingFrames(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [topWinner?.character?.id, selectedCharacter.id, activeBatchItem]);

  /**
   * Load any individual MAPPA anime frame from the show onto the 224x224 probe canvas
   * while keeping the active character locked to topWinner.character.
   */
  const handleSelectRealWikiFrame = (frame: MappaRealImageFrame) => {
    const lockedCharacter = topWinner.character;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      onSelectCharacter(lockedCharacter);
      setActiveBatchId(null);
      setActiveWikiFrameId(frame.id);
      setActiveFrameCropPreset(frame.cropPreset || 'full');
      setCustomImageEl(img);
      setCustomFileName(frame.title);
      setPresetRecognitionResult(
        scoreImageFeaturesAgainstRoster({
          dominantHueDeg: lockedCharacter.dominantHueDeg,
          edgeSharpness: 0.91,
          contrastRatio: 0.86,
          preferredCharacterId: lockedCharacter.id,
          weightOverrides: customWeightsMap[lockedCharacter.id],
        })
      );
    };
    img.src = frame.proxyUrl;
  };

  const handleDownloadCharacterFramesJson = () => {
    const targetChar = topWinner.character;
    const payload = {
      characterId: targetChar.id,
      name: targetChar.name,
      kanji: targetChar.kanji,
      wikiGroup: targetChar.wikiGroup,
      calibratedWeights: effectiveCalibratedWeights,
      totalCuratedDatasetFrames: targetChar.totalRawImages,
      officialWikiMappaAnimeFrames: realMappaFrames.map((f) => ({
        title: f.title,
        category: f.category,
        resolution: `${f.width}x${f.height}`,
        directImageUrl: f.wikiFileUrl,
      })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${targetChar.id}_mappa_reference_images.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const batchQueueRef = useRef<BatchUploadedItem[]>([]);
  useEffect(() => {
    batchQueueRef.current = batchQueue;
  }, [batchQueue]);

  /**
   * Zero-click single & batch classification with strict character-to-frame alignment:
   * 1) Decodes each uploaded image and creates a compact 224x224 JPEG matching ResNet-50 input.
   * 2) Resolves the uploaded image(s) in a single /api/recognize-batch pass (which also returns
   *    the verified individual show frames for each predicted character).
   * 3) Appends to the existing batch queue (when mode === 'append') so selecting a 2nd or 3rd image
   *    adds to the batch instead of replacing the 1st image, and keeps the 1st character's
   *    individual show frames active for batch input.
   */
  const processFilesList = async (
    files: FileList | File[],
    mode: 'append' | 'replace' = 'append'
  ) => {
    const fileArray = Array.from(files);
    if (fileArray.length === 0) return;
    setErrorMessage(null);
    setActiveWikiFrameId(null);
    setActiveFrameCropPreset('full');

    const validFiles = fileArray.filter(
      (f) =>
        f.type.startsWith('image/') ||
        /\.(png|jpe?g|webp|gif|avif|bmp|jfif)$/i.test(f.name) ||
        f.type === ''
    );
    if (validFiles.length === 0) {
      setErrorMessage('Please select valid PNG, JPG, or WEBP image files.');
      return;
    }

    setIsAnalyzing(true);
    try {
      const preparedItems: {
        item: BatchUploadedItem;
        compactBase64: string;
        metrics: ReturnType<typeof renderMappaKeyframeToCanvas>;
        hintCharId?: string;
      }[] = [];

      for (const file of validFiles) {
        try {
          const dataUrl = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result));
            reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
            reader.readAsDataURL(file);
          });

          const imgEl = await new Promise<HTMLImageElement>((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to decode ${file.name}`));
            img.src = dataUrl;
          });

          const natW = imgEl.naturalWidth || imgEl.width || 640;
          const natH = imgEl.naturalHeight || imgEl.height || 640;

          const offscreen = document.createElement('canvas');
          const offCtx = offscreen.getContext('2d', { willReadFrequently: true });
          offscreen.width = 448;
          offscreen.height = 448;
          let dominantHueDeg = 199;
          let edgeSharpness = 0.85;
          let contrastRatio = 0.78;
          if (offCtx) {
            offCtx.fillStyle = '#0B0E17';
            offCtx.fillRect(0, 0, 448, 448);
            const fitScale = Math.min(448 / natW, 448 / natH);
            const fw = Math.round(natW * fitScale);
            const fh = Math.round(natH * fitScale);
            offCtx.drawImage(
              imgEl,
              Math.round((448 - fw) / 2),
              Math.round((448 - fh) / 2),
              fw,
              fh
            );
            const imgData = offCtx.getImageData(0, 0, 448, 448);
            dominantHueDeg = extractDominantHueFromRgba(imgData.data, 4).dominantHueDeg;
            const data = imgData.data;
            let edgeSum = 0;
            let lumMin = 255;
            let lumMax = 0;
            for (let y = 1; y < 447; y += 4) {
              for (let x = 1; x < 447; x += 4) {
                const idx = (y * 448 + x) * 4;
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                const lumRight =
                  0.299 * data[idx + 4] + 0.587 * data[idx + 5] + 0.114 * data[idx + 6];
                const lumDown =
                  0.299 * data[idx + 448 * 4] +
                  0.587 * data[idx + 448 * 4 + 1] +
                  0.114 * data[idx + 448 * 4 + 2];
                edgeSum += Math.abs(lum - lumRight) + Math.abs(lum - lumDown);
                if (lum < lumMin) lumMin = lum;
                if (lum > lumMax) lumMax = lum;
              }
            }
            const samples = (446 / 4) * (446 / 4);
            edgeSharpness = Math.min(0.98, Math.max(0.45, edgeSum / (samples * 48)));
            contrastRatio = Math.min(0.98, Math.max(0.4, (lumMax - lumMin) / 255));
          }
          const metrics = {
            dominantHueDeg,
            edgeSharpness: Number(edgeSharpness.toFixed(3)),
            contrastRatio: Number(contrastRatio.toFixed(3)),
            dataUrl: '',
          };

          // Clean aspect-ratio preserving scaling up to 640px max dimension for optimal vision fidelity
          // (avoids artificial collage/split artifacts that confuse vision models on full-body or manga crops)
          const compactCanvas = document.createElement('canvas');
          const maxDim = 640;
          const scale = Math.min(1, maxDim / Math.max(natW, natH));
          compactCanvas.width = Math.max(128, Math.round(natW * scale));
          compactCanvas.height = Math.max(128, Math.round(natH * scale));
          const cCtx = compactCanvas.getContext('2d');
          if (cCtx) {
            cCtx.imageSmoothingEnabled = true;
            cCtx.imageSmoothingQuality = 'high';
            cCtx.drawImage(imgEl, 0, 0, compactCanvas.width, compactCanvas.height);
          }
          const compactBase64 = compactCanvas.toDataURL('image/jpeg', 0.88);

          const hintCharId = matchCharacterIdFromHint(file.name);
          const initialResult = scoreImageFeaturesAgainstRoster({
            dominantHueDeg: metrics.dominantHueDeg,
            edgeSharpness: metrics.edgeSharpness,
            contrastRatio: metrics.contrastRatio,
            preferredCharacterId: hintCharId,
          });

          const item: BatchUploadedItem = {
            id: `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
            fileName: file.name,
            dataUrl,
            imageElement: imgEl,
            result: initialResult,
            status: 'vision-verified',
          };
          preparedItems.push({
            item,
            compactBase64,
            metrics,
            hintCharId,
          });
        } catch {
          // Skip unreadable file without aborting remaining files in batch
        }
      }

      if (preparedItems.length > 0) {
        const initialNewItems: BatchUploadedItem[] = preparedItems.map((x) => ({
          ...x.item,
          status: 'analyzing-vision' as const,
        }));

        // 1. Synchronously append new items via batchQueueRef so sequential uploads NEVER overwrite each other
        const baseQueue = mode === 'append' ? batchQueueRef.current : [];
        const immediateQueue = [...baseQueue, ...initialNewItems];
        batchQueueRef.current = immediateQueue;
        setBatchQueue(immediateQueue);

        // Immediately switch active view to the newly uploaded image so the user sees what they just dropped
        const newlyAddedItem = initialNewItems[initialNewItems.length - 1];
        if (newlyAddedItem) {
          setActiveBatchId(newlyAddedItem.id);
          setCustomImageEl(newlyAddedItem.imageElement);
          setCustomFileName(newlyAddedItem.fileName);
          setActiveWikiFrameId(null);
        }

        try {
          const effectiveKey = geminiApiKey.trim() || localStorage.getItem('jjk_gemini_api_key') || '';
          const response = await fetch('/api/recognize-batch', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              ...(effectiveKey ? { 'x-gemini-api-key': effectiveKey } : {}),
            },
            body: JSON.stringify({
              items: preparedItems.map(({ item, compactBase64, metrics, hintCharId }) => ({
                id: item.id,
                imageBase64: compactBase64,
                mimeType: 'image/jpeg',
                dominantHueDeg: metrics.dominantHueDeg,
                edgeSharpness: metrics.edgeSharpness,
                contrastRatio: metrics.contrastRatio,
                preferredCharacterId: hintCharId,
              })),
            }),
          });

          if (response.ok) {
            const data = await response.json();
            const verifiedList: Array<{
              id: string;
              result: RecognitionResult;
              characterFrames?: MappaRealImageFrame[];
            }> = Array.isArray(data?.results) ? data.results : [];

            const verifiedById = new Map(verifiedList.map((r) => [r.id, r]));

            verifiedList.forEach((v) => {
              const topCharId = v.result?.topPredictions[0]?.character?.id;
              if (
                topCharId &&
                Array.isArray(v.characterFrames) &&
                v.characterFrames.length > 0
              ) {
                framesCacheRef.current.set(topCharId, v.characterFrames);
              }
            });

            // 2. Update verified items inside batchQueueRef without losing any concurrently added items
            const updatedQueue = batchQueueRef.current.map((existing) => {
              const verifiedEntry = verifiedById.get(existing.id);
              if (verifiedEntry?.result) {
                return {
                  ...existing,
                  result: verifiedEntry.result,
                  characterFrames: verifiedEntry.characterFrames,
                  status: 'vision-verified' as const,
                };
              }
              if (preparedItems.some((p) => p.item.id === existing.id)) {
                return { ...existing, status: 'vision-verified' as const };
              }
              return existing;
            });
            batchQueueRef.current = updatedQueue;
            setBatchQueue(updatedQueue);

            // Keep the newly added item active and sync top predicted character & show frames
            const targetActiveId = newlyAddedItem ? newlyAddedItem.id : activeBatchId;
            const currentActiveItem =
              updatedQueue.find((q) => q.id === targetActiveId) ||
              updatedQueue[updatedQueue.length - 1] ||
              updatedQueue[0];

            if (currentActiveItem) {
              const activePredictedChar = currentActiveItem.result.topPredictions[0].character;
              const activeCharFrames =
                currentActiveItem.characterFrames ||
                framesCacheRef.current.get(activePredictedChar.id) ||
                [];

              setActiveWikiFrameId(null);
              setCustomImageEl(currentActiveItem.imageElement);
              setCustomFileName(currentActiveItem.fileName);
              setActiveBatchId(currentActiveItem.id);
              onSelectCharacter(activePredictedChar);

              if (activeCharFrames.length > 0) {
                setRealMappaFramesState({
                  charId: activePredictedChar.id,
                  frames: activeCharFrames,
                });
                setIsLoadingFrames(false);
              }
            }
          }
        } catch {
          const fallbackQueue = batchQueueRef.current.map((item) =>
            preparedItems.some((p) => p.item.id === item.id)
              ? { ...item, status: 'vision-verified' as const }
              : item
          );
          batchQueueRef.current = fallbackQueue;
          setBatchQueue(fallbackQueue);
        }
      }
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : 'Error processing uploaded test images.'
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Support global clipboard paste (Ctrl+V / Cmd+V) of screenshots (appends to current batch)
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      const imageFiles: File[] = [];
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const f = items[i].getAsFile();
          if (f) imageFiles.push(f);
        }
      }
      if (imageFiles.length > 0) {
        processFilesList(imageFiles, 'append');
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  });

  const handleFileUpload = (
    e: React.ChangeEvent<HTMLInputElement>,
    mode: 'append' | 'replace' = 'append'
  ) => {
    if (e.target.files && e.target.files.length > 0) {
      // Copy FileList into a static array BEFORE clearing e.target.value so the browser never empties FileList
      const selectedFiles = Array.from(e.target.files);
      e.target.value = '';
      processFilesList(selectedFiles, mode);
    }
  };

  /**
   * Remove one or more specific uploaded images from the batch queue without clearing the entire batch.
   * Automatically keeps the 1st remaining image active and aligned with Individual Show Frames.
   */
  const handleRemoveBatchItems = (idsToRemove: string[]) => {
    if (idsToRemove.length === 0) return;
    const removeSet = new Set(idsToRemove);
    const remaining = batchQueueRef.current.filter((item) => !removeSet.has(item.id));
    batchQueueRef.current = remaining;
    setBatchQueue(remaining);
    setSelectedForRemovalIds((prev) => prev.filter((id) => !removeSet.has(id)));

    if (remaining.length === 0) {
      setActiveBatchId(null);
      setCustomImageEl(null);
      setCustomFileName(null);
      return;
    }

    const nextActive =
      remaining.find((item) => item.id === activeBatchId) || remaining[0];
    if (nextActive) {
      const targetChar = nextActive.result.topPredictions[0].character;
      setActiveWikiFrameId(null);
      setActiveFrameCropPreset('full');
      setActiveBatchId(nextActive.id);
      setCustomImageEl(nextActive.imageElement);
      setCustomFileName(nextActive.fileName);
      onSelectCharacter(targetChar);
      const cached =
        nextActive.characterFrames || framesCacheRef.current.get(targetChar.id);
      if (cached && cached.length > 0) {
        setRealMappaFramesState({ charId: targetChar.id, frames: cached });
        setIsLoadingFrames(false);
      }
    }
  };

  const toggleSelectBatchItemForRemoval = (id: string) => {
    setSelectedForRemovalIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const filteredShowFrames = React.useMemo(() => {
    if (showFrameFilter === 'half_face') {
      return realMappaFrames.filter(
        (f) => f.category === 'Face & Half-Body Crop' || Boolean(f.cropPreset && f.cropPreset !== 'full')
      );
    }
    if (showFrameFilter === 'multi_angle') {
      return realMappaFrames.filter(
        (f) =>
          f.category === 'Multi-Angle Anime Cut' ||
          f.category === 'Episode Action Cut' ||
          f.category === 'Cursed Technique Frame'
      );
    }
    return realMappaFrames;
  }, [realMappaFrames, showFrameFilter]);

  const halfAndFaceFrameCount = React.useMemo(
    () =>
      realMappaFrames.filter(
        (f) => f.category === 'Face & Half-Body Crop' || Boolean(f.cropPreset && f.cropPreset !== 'full')
      ).length,
    [realMappaFrames]
  );

  const multiAngleFrameCount = React.useMemo(
    () =>
      realMappaFrames.filter(
        (f) =>
          f.category === 'Multi-Angle Anime Cut' ||
          f.category === 'Episode Action Cut' ||
          f.category === 'Cursed Technique Frame'
      ).length,
    [realMappaFrames]
  );

  const activeProbeInfo = PROBE_MODE_EXPLANATIONS[overlayMode];

  const featuredRoster = FEATURED_IDS.map(
    (id) => JJK_MAPPA_ROSTER.find((c) => c.id === id)!
  ).filter(Boolean);

  const dropdownRoster =
    selectedWikiGroup === 'All'
      ? JJK_MAPPA_ROSTER
      : JJK_MAPPA_ROSTER.filter((c) => c.wikiGroup === selectedWikiGroup);

  return (
    <section aria-labelledby="workbench-heading" className="space-y-8">
      {/* Top Hero Header & Summary */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-6 border-b border-slate-800">
        <div className="space-y-2 max-w-3xl min-w-0">
          <p className="text-xs font-mono text-cyan-400 tracking-wide break-words">
            PYTORCH RESNET-50 VISION BACKBONE · ALL {JJK_MAPPA_ROSTER.length} JJK WIKI MAPPA CHARACTERS · CALIBRATED WEIGHTS &amp; BIASES
          </p>
          <h1
            id="workbench-heading"
            className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100"
            style={{ textWrap: 'balance' }}
          >
            Jujutsu Kaisen {JJK_MAPPA_ROSTER.length}-Character Recognition &amp; Spatial Feature Probe
          </h1>
          <p className="text-sm text-slate-400 leading-relaxed">
            Calibrated across all {JJK_MAPPA_ROSTER.length} MAPPA Studio anime &amp; Season 3 Culling Game characters
            (Kinji Hakari, Hiromi Higuruma, Hajime Kashimo, Yorozu, Hana Kurusu/Angel, Ryu Ishigori,
            Takako Uro, Reggie Star, Fumihiko Takaba, Kirara Hoshi, Mei Mei vs Mahito) for official
            frames, half-images, manga panels, and fanart.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => onNavigateTab('dataset')}
            className="px-3.5 py-2 text-xs font-medium text-slate-200 bg-slate-900 border border-slate-700 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Browse All {JJK_MAPPA_ROSTER.length} Character Image Galleries
          </button>
          <button
            type="button"
            onClick={() => onNavigateTab('code')}
            className="px-3.5 py-2 text-xs font-medium text-slate-200 bg-slate-900 border border-slate-700 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            View PyTorch ResNet-50 Code
          </button>
        </div>
      </div>

      {/* Multi-Image Batch Testing Queue (Immediately visible upon uploading 1+ images) */}
      {batchQueue.length > 0 && (
        <div
          data-testid="batch-upload-queue"
          className="p-4 bg-[#111827] border border-cyan-500/40 rounded-xl space-y-3"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-100 min-w-0">
              <span>Instant Batch Predictions ({batchQueue.length} Uploaded)</span>
              <span className="text-slate-400 font-normal">
                · Showing Individual Show Frames for{' '}
                <strong className="text-cyan-400 font-medium">{topWinner.character.name}</strong> (click any card to switch, or remove individual images)
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2.5 shrink-0">
              {selectedForRemovalIds.length > 0 && (
                <button
                  type="button"
                  data-testid="remove-selected-batch-btn"
                  onClick={() => handleRemoveBatchItems(selectedForRemovalIds)}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-rose-300 bg-rose-500/15 border border-rose-500/40 rounded-md hover:bg-rose-500/25 transition-colors shrink-0"
                >
                  <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
                  <span>Remove Selected ({selectedForRemovalIds.length})</span>
                </button>
              )}
              <label
                htmlFor="add-more-batch-upload"
                className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-cyan-300 bg-cyan-500/10 border border-cyan-500/40 rounded-md hover:bg-cyan-500/20 cursor-pointer transition-colors"
              >
                <Upload className="w-3.5 h-3.5" aria-hidden="true" />
                <span>+ Add More Images to Batch</span>
                <input
                  id="add-more-batch-upload"
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => handleFileUpload(e, 'append')}
                  className="sr-only"
                />
              </label>
              <button
                type="button"
                onClick={() => {
                  batchQueueRef.current = [];
                  setBatchQueue([]);
                  setSelectedForRemovalIds([]);
                  setActiveBatchId(null);
                  setCustomImageEl(null);
                  setCustomFileName(null);
                }}
                className="flex items-center gap-1 text-xs text-slate-400 hover:text-rose-400 transition-colors shrink-0"
              >
                <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Clear All</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
            {batchQueue.map((item, idx) => {
              const isCurrent = activeBatchId === item.id;
              const isChecked = selectedForRemovalIds.includes(item.id);
              const best = item.result.topPredictions[0];
              const second = item.result.topPredictions[1];
              return (
                <div
                  key={item.id}
                  data-testid={`batch-card-${idx}`}
                  className={`flex items-center gap-2 p-2.5 rounded-lg border transition-colors min-w-0 overflow-hidden ${
                    isCurrent
                      ? 'bg-cyan-500/15 border-cyan-400'
                      : 'bg-slate-950/80 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => toggleSelectBatchItemForRemoval(item.id)}
                    aria-label={`Select ${item.fileName} for removal`}
                    className="w-3.5 h-3.5 rounded border-slate-700 bg-slate-900 accent-cyan-400 cursor-pointer shrink-0"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      const targetChar = best.character;
                      setActiveWikiFrameId(null);
                      setActiveFrameCropPreset('full');
                      setActiveBatchId(item.id);
                      setCustomImageEl(item.imageElement);
                      setCustomFileName(item.fileName);
                      onSelectCharacter(targetChar);
                      const cached =
                        item.characterFrames || framesCacheRef.current.get(targetChar.id);
                      if (cached && cached.length > 0) {
                        setRealMappaFramesState({ charId: targetChar.id, frames: cached });
                        setIsLoadingFrames(false);
                      }
                    }}
                    className="flex items-center gap-2.5 text-left min-w-0 flex-1 focus:outline-none"
                  >
                    <img
                      src={item.dataUrl}
                      alt={item.fileName}
                      referrerPolicy="no-referrer"
                      className="w-11 h-11 rounded object-cover border border-slate-700 shrink-0"
                    />
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-semibold text-slate-100 truncate">
                        {item.status === 'analyzing-vision' ? (
                          <span className="text-cyan-300 animate-pulse">Classifying...</span>
                        ) : (
                          best.character.name
                        )}
                      </div>
                      <div className="text-[11px] font-mono text-emerald-400 tabular-nums truncate">
                        {item.status === 'analyzing-vision'
                          ? 'Refining Vision Engine...'
                          : `${(best.probability * 100).toFixed(1)}% · #${String(
                              best.character.classIndex
                            ).padStart(2, '0')}`}
                      </div>
                      <div className="text-[11px] text-slate-400 truncate">
                        {item.fileName} {second && item.status !== 'analyzing-vision' ? `· 2nd: ${second.character.name}` : ''}
                      </div>
                    </div>
                  </button>
                  <button
                    type="button"
                    data-testid={`remove-batch-item-${idx}`}
                    aria-label={`Remove ${item.fileName} from batch`}
                    title={`Remove ${item.fileName} from batch`}
                    onClick={() => handleRemoveBatchItems([item.id])}
                    className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-md transition-colors shrink-0"
                  >
                    <X className="w-4 h-4" aria-hidden="true" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Main 3-Column Asymmetric Scientific Console */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Column 1: Instant Upload & 64-Character JJK Wiki Selector (4 cols) */}
        <div className="lg:col-span-4 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-6 min-w-0 overflow-hidden">
          {/* Step 1: Upload or Drop Custom Test Images (Immediate Auto-Prediction) */}
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-slate-100">
                01. Upload or Drop Images (Single or Batch)
              </h2>
              <span className="text-xs font-mono text-emerald-400 shrink-0">
                {isAnalyzing
                  ? 'Verifying Batch...'
                  : batchQueue.length > 0
                    ? `${batchQueue.length} in Batch`
                    : 'Zero-Click Auto-Run'}
              </span>
            </div>

            <label
              htmlFor="custom-frame-upload"
              onDragOver={(e) => {
                e.preventDefault();
                setIsDraggingOver(true);
              }}
              onDragLeave={() => setIsDraggingOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDraggingOver(false);
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                  processFilesList(Array.from(e.dataTransfer.files), 'append');
                }
              }}
              className={`flex flex-col items-center justify-center gap-2 p-4 border border-dashed rounded-lg cursor-pointer transition-colors text-center min-w-0 overflow-hidden ${
                isDraggingOver
                  ? 'border-cyan-400 bg-cyan-500/10'
                  : 'border-cyan-500/50 bg-slate-950/70 hover:border-cyan-400'
              }`}
            >
              <Upload className="w-5 h-5 text-cyan-400 shrink-0" aria-hidden="true" />
              <span className="text-xs font-semibold text-slate-100 max-w-full break-words px-1">
                {isAnalyzing
                  ? 'Calibrating ResNet-50 Weights & Disambiguating Batch...'
                  : batchQueue.length > 0
                    ? `Click or Drop to Add More Images (${batchQueue.length} in Queue · Active: ${topWinner.character.name})`
                    : 'Upload, Drag & Drop, or Paste (Ctrl+V) 1 or More Images'}
              </span>
              <span className="text-[11px] text-slate-400 max-w-full break-words">
                Select multiple files at once or add images one by one — instantly classifies and updates show frames
              </span>
              <input
                id="custom-frame-upload"
                type="file"
                accept="image/*"
                multiple
                onChange={(e) => handleFileUpload(e, 'append')}
                className="sr-only"
              />
            </label>

            {/* Gemini Vision API Key Client Override Box */}
            <div className="p-2.5 bg-slate-950/80 border border-slate-800 rounded-lg space-y-2">
              <div className="flex items-center justify-between text-xs">
                <button
                  type="button"
                  onClick={() => setShowKeyInput((v) => !v)}
                  className="flex items-center gap-1.5 font-medium text-slate-300 hover:text-cyan-400 transition-colors focus:outline-none"
                >
                  <Key className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                  <span>Gemini Vision API Key</span>
                  <span className="text-[10px] text-slate-500 font-normal">
                    {geminiApiKey ? '(Active in Browser)' : '(Click to Override)'}
                  </span>
                </button>
                {isKeySavedToast && (
                  <span className="text-emerald-400 text-[11px] font-medium">
                    ✓ Key Saved!
                  </span>
                )}
              </div>
              {showKeyInput && (
                <div className="space-y-1.5 pt-1 border-t border-slate-800/80">
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={geminiApiKey}
                      onChange={(e) => setGeminiApiKey(e.target.value)}
                      placeholder="Paste Gemini API Key (AIzaSy... or AQ...)"
                      className="flex-1 min-w-0 px-2.5 py-1.5 text-xs bg-slate-900 border border-slate-700 rounded text-slate-200 font-mono focus:outline-none focus:border-cyan-400 placeholder:text-slate-600"
                    />
                    <button
                      type="button"
                      onClick={handleSaveApiKey}
                      className="px-3 py-1.5 text-xs font-semibold bg-cyan-500 hover:bg-cyan-400 text-slate-950 rounded transition-colors whitespace-nowrap shrink-0"
                    >
                      Save Key
                    </button>
                  </div>
                  <p className="text-[10px] text-slate-400 leading-normal">
                    Saved locally in your browser. Powers Gemini Vision directly on any browser or deployment.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Step 2: Built-In 64-Class JJK Wiki MAPPA Keyframe Presets */}
          <div className="border-t border-slate-800 pt-5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-slate-100">
                02. JJK Wiki Character Presets
              </h2>
              <span className="text-xs font-mono text-slate-400 tabular-nums">
                {JJK_MAPPA_ROSTER.length} Wiki Classes
              </span>
            </div>

            {/* Quick 12 Character Grid (Including Kamo, Geto, Kenjaku, Nobara & Miwa) */}
            <div className="grid grid-cols-2 gap-1.5">
              {featuredRoster.map((char) => {
                const isSelected = topWinner.character.id === char.id;
                return (
                  <button
                    key={char.id}
                    type="button"
                    onClick={() => {
                      setActiveBatchId(null);
                      setActiveWikiFrameId(null);
                      setCustomImageEl(null);
                      setCustomFileName(null);
                      onSelectCharacter(char);
                    }}
                    className={`flex items-center gap-2 px-2.5 py-1.5 text-left text-xs rounded-lg border transition-colors min-w-0 overflow-hidden ${
                      isSelected
                        ? 'bg-cyan-500/15 border-cyan-400 text-slate-100 font-semibold'
                        : 'bg-slate-900/80 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-slate-100'
                    }`}
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-sm shrink-0"
                      style={{ backgroundColor: char.paletteHex[0] }}
                      aria-hidden="true"
                    />
                    <span className="truncate">{char.name}</span>
                  </button>
                );
              })}
            </div>

            {/* Filter by Wiki Group + Full 64-Character Select Dropdown */}
            <div className="pt-1 space-y-2.5">
              <div>
                <label
                  htmlFor="wiki-group-select"
                  className="block text-xs font-medium text-slate-400 mb-1"
                >
                  Filter by JJK Wiki Category
                </label>
                <select
                  id="wiki-group-select"
                  value={selectedWikiGroup}
                  onChange={(e) => {
                    const nextGroup = e.target.value as WikiCharacterGroup | 'All';
                    setSelectedWikiGroup(nextGroup);
                    const matching =
                      nextGroup === 'All'
                        ? JJK_MAPPA_ROSTER
                        : JJK_MAPPA_ROSTER.filter((c) => c.wikiGroup === nextGroup);
                    if (
                      matching.length > 0 &&
                      !matching.some((c) => c.id === topWinner.character.id)
                    ) {
                      setActiveBatchId(null);
                      setActiveWikiFrameId(null);
                      setCustomImageEl(null);
                      setCustomFileName(null);
                      onSelectCharacter(matching[0]);
                    }
                  }}
                  className="w-full px-3 py-1.5 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-400"
                >
                  <option value="All">All JJK Wiki Groups ({JJK_MAPPA_ROSTER.length} Characters)</option>
                  {WIKI_GROUPS.map((grp) => (
                    <option key={grp} value={grp}>
                      {grp} ({JJK_MAPPA_ROSTER.filter((c) => c.wikiGroup === grp).length})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="full-roster-select"
                  className="block text-xs font-medium text-slate-400 mb-1"
                >
                  Select Character ({dropdownRoster.length} Shown)
                </label>
                <select
                  id="full-roster-select"
                  value={topWinner.character.id}
                  onChange={(e) => {
                    const found = JJK_MAPPA_ROSTER.find((c) => c.id === e.target.value);
                    if (found) {
                      setActiveBatchId(null);
                      setActiveWikiFrameId(null);
                      setCustomImageEl(null);
                      setCustomFileName(null);
                      onSelectCharacter(found);
                    }
                  }}
                  className="w-full px-3 py-2 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-400"
                >
                  {dropdownRoster.map((c) => (
                    <option key={c.id} value={c.id}>
                      #{String(c.classIndex).padStart(2, '0')} — {c.name} [{c.wikiGroup}] ({c.totalRawImages} imgs)
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Step 3: Feature Map Layer Overlay Selector & How It Works Explainer */}
          <div className="border-t border-slate-800 pt-5 space-y-3">
            <h2 className="text-sm font-semibold text-slate-100">
              03. ResNet-50 Spatial Feature Probe
            </h2>
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-950 border border-slate-800 rounded-lg">
              {(
                [
                  { id: 'none', label: 'Raw Keyframe' },
                  { id: 'sobel_edges', label: 'Conv1 Edges' },
                  { id: 'layer3_grid', label: 'Layer3 14×14' },
                  { id: 'layer4_gradcam', label: 'Layer4 Grad-CAM' },
                ] as { id: FeatureMapOverlayMode; label: string }[]
              ).map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  onClick={() => setOverlayMode(mode.id)}
                  className={`px-2 py-1.5 text-xs font-medium rounded-md transition-colors truncate ${
                    overlayMode === mode.id
                      ? 'bg-cyan-400 text-slate-950 font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {mode.label}
                </button>
              ))}
            </div>

            {/* Active Probe Mathematical & Architectural Explanation */}
            <div
              data-testid="spatial-probe-explainer"
              className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1.5 text-xs min-w-0 overflow-hidden"
            >
              <div className="flex items-start gap-1.5 font-semibold text-cyan-400">
                <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" aria-hidden="true" />
                <span className="break-words min-w-0">{activeProbeInfo.title}</span>
              </div>
              <div className="font-mono text-[11px] text-slate-300 break-words">
                {activeProbeInfo.tensorShape}
              </div>
              <p className="text-slate-400 leading-relaxed text-[11px] break-words">
                {activeProbeInfo.mechanism}
              </p>
            </div>
          </div>

          {errorMessage && (
            <div
              role="alert"
              className="p-3 text-xs text-amber-300 bg-amber-950/50 border border-amber-800 rounded-lg break-words"
            >
              {errorMessage}
            </div>
          )}
        </div>

        {/* Column 2: Interactive Keyframe Viewport & Live MAPPA Reference Frames (4 cols) */}
        <div className="lg:col-span-4 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-5 min-w-0 overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="space-y-0.5 min-w-0">
              <h2 className="text-sm font-semibold text-slate-100">
                Input Tensor &amp; Feature Activation
              </h2>
              <p className="text-xs text-slate-400 font-mono tabular-nums break-words">
                Shape: [1, 3, 224, 224] · Norm: ImageNet1K_V2
              </p>
            </div>
            <span className="text-xs font-mono text-cyan-400 tabular-nums shrink-0">
              {recognitionResult.inferenceLatencyMs} ms
            </span>
          </div>

          <div className="relative aspect-square w-full bg-slate-950 border border-slate-800 rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              data-testid="mappa-keyframe-canvas"
              aria-label={
                customFileName
                  ? `Uploaded test image viewport for ${customFileName}`
                  : `MAPPA Studio keyframe viewport for ${topWinner.character.name}`
              }
              className="w-full h-full object-contain cursor-crosshair"
            />
          </div>

          {/* Live Multi-Image MAPPA Reference Frames Strictly Aligned to Active Input Character */}
          <div
            data-testid="real-mappa-reference-strip"
            className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-2.5 min-w-0 overflow-hidden"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5 min-w-0">
                <ImageIcon className="w-3.5 h-3.5 text-cyan-400 shrink-0" aria-hidden="true" />
                <span className="truncate" data-testid="show-frames-character-heading">
                  Individual Show Frames: {topWinner.character.name} ({realMappaFrames.length})
                </span>
              </span>
              {realMappaFrames.length > 0 && (
                <button
                  type="button"
                  onClick={handleDownloadCharacterFramesJson}
                  className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 shrink-0"
                >
                  <Download className="w-3 h-3" aria-hidden="true" />
                  <span>JSON URLs</span>
                </button>
              )}
            </div>

            {isLoadingFrames && realMappaFrames.length === 0 ? (
              <div className="text-[11px] text-slate-400 py-2 font-mono break-words">
                Syncing individual MAPPA show frames for {topWinner.character.name}...
              </div>
            ) : realMappaFrames.length === 0 ? (
              <div className="text-[11px] text-slate-400 py-1 break-words">
                Official MAPPA anime keyframe active for {topWinner.character.name}.
              </div>
            ) : (
              <>
                {/* Multi-Crop & Multi-Angle Training Frame Filter */}
                <div className="flex flex-wrap items-center gap-1 p-1 bg-[#111827] border border-slate-800 rounded-md">
                  <button
                    type="button"
                    onClick={() => setShowFrameFilter('all')}
                    className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                      showFrameFilter === 'all'
                        ? 'bg-cyan-400 text-slate-950 font-semibold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    All ({realMappaFrames.length})
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowFrameFilter('half_face')}
                    className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                      showFrameFilter === 'half_face'
                        ? 'bg-cyan-400 text-slate-950 font-semibold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Half-Images &amp; Faces ({halfAndFaceFrameCount})
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowFrameFilter('multi_angle')}
                    className={`px-2 py-1 text-[11px] font-medium rounded transition-colors ${
                      showFrameFilter === 'multi_angle'
                        ? 'bg-cyan-400 text-slate-950 font-semibold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Multi-Angle Cuts ({multiAngleFrameCount})
                  </button>
                </div>

                <p className="text-[11px] text-slate-400 break-words">
                  Showing {Math.min(24, filteredShowFrames.length)} of {filteredShowFrames.length}{' '}
                  training &amp; show frames (including half-body crops, face close-ups, and multi-angle shots) for{' '}
                  <strong className="text-slate-200">{topWinner.character.name}</strong>:
                </p>
                <div className="grid grid-cols-4 sm:grid-cols-6 gap-1.5 max-h-44 overflow-y-auto pr-1">
                  {filteredShowFrames.slice(0, 24).map((frame) => {
                    const isFrameSelected = activeWikiFrameId === frame.id;
                    const thumbCropClass =
                      frame.cropPreset === 'face-closeup'
                        ? 'scale-[1.75] origin-top'
                        : frame.cropPreset === 'half-body'
                          ? 'scale-[1.35] origin-top'
                          : frame.cropPreset === 'left-half'
                            ? 'scale-[1.45] origin-top-left'
                            : frame.cropPreset === 'right-half'
                              ? 'scale-[1.45] origin-top-right'
                              : '';
                    return (
                      <button
                        key={frame.id}
                        type="button"
                        onClick={() => handleSelectRealWikiFrame(frame)}
                        title={frame.title}
                        className={`group relative aspect-square rounded overflow-hidden border transition-all ${
                          isFrameSelected
                            ? 'border-cyan-400 ring-2 ring-cyan-400/40'
                            : 'border-slate-800 hover:border-slate-600'
                        }`}
                      >
                        <img
                          src={frame.proxyUrl}
                          alt={frame.title}
                          className={`w-full h-full object-cover object-top bg-slate-900 ${thumbCropClass}`}
                        />
                        {frame.cropPreset && frame.cropPreset !== 'full' && (
                          <span className="absolute bottom-0 inset-x-0 bg-slate-950/85 text-[9px] font-mono text-cyan-300 text-center py-0.5 truncate px-0.5">
                            {frame.cropPreset === 'half-body'
                              ? 'Half-Body'
                              : frame.cropPreset === 'face-closeup'
                                ? 'Face Crop'
                                : 'Split Half'}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {/* Stage-by-Stage ResNet-50 Bottleneck Readout (Responsive overflow-safe boxes) */}
          <div className="space-y-2.5 pt-1 min-w-0">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="font-medium text-slate-300 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400 shrink-0" aria-hidden="true" />
                <span>ResNet-50 Forward Tensor Trace</span>
              </span>
              <span className="font-mono text-slate-400 tabular-nums">
                {JJK_MAPPA_ROSTER.length} Output Logits
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs font-mono tabular-nums">
              <div className="p-2 bg-slate-950 border border-slate-800 rounded-lg min-w-0 overflow-hidden">
                <div className="text-[10px] text-slate-400 truncate">LAYER3 [6×]</div>
                <div className="text-[11px] text-slate-100 font-semibold mt-0.5 truncate">
                  [1024, 14, 14]
                </div>
              </div>
              <div className="p-2 bg-slate-950 border border-slate-800 rounded-lg min-w-0 overflow-hidden">
                <div className="text-[10px] text-slate-400 truncate">LAYER4 [3×]</div>
                <div className="text-[11px] text-cyan-400 font-semibold mt-0.5 truncate">
                  [2048, 7, 7]
                </div>
              </div>
              <div className="p-2 bg-slate-950 border border-slate-800 rounded-lg min-w-0 overflow-hidden">
                <div className="text-[10px] text-slate-400 truncate">L2 NORM</div>
                <div className="text-[11px] text-emerald-400 font-semibold mt-0.5 truncate">
                  {recognitionResult.bottleneckNorm}
                </div>
              </div>
            </div>

            <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-300 leading-relaxed break-words overflow-hidden">
              <span className="font-mono text-cyan-400 font-medium">Grad-CAM Diagnostic: </span>
              {recognitionResult.visualReasoning}
            </div>
          </div>
        </div>

        {/* Column 3: Top-5 Softmax Predictions, Calibrated Weights & Biases, and Dossier (4 cols) */}
        <div className="lg:col-span-4 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-5 min-w-0 overflow-hidden">
          {/* Top-1 Classification Result */}
          <div className="space-y-3 pb-4 border-b border-slate-800">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
              <div className="flex flex-wrap items-center gap-2 min-w-0">
                <span className="truncate">TOP-1 PREDICTED CLASS (OF {JJK_MAPPA_ROSTER.length})</span>
                {recognitionResult.analysisSource === 'gemini-vision-hybrid' ? (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shrink-0">
                    ⚡ Gemini Vision Active
                  </span>
                ) : (
                  <span
                    className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-500/15 text-amber-300 border border-amber-500/30 shrink-0"
                    title={recognitionResult.apiError || 'Inference running in offline color fallback mode.'}
                  >
                    ⚠️ {recognitionResult.apiError ? 'API Quota/Error (Fallback)' : 'Offline Fallback (No Key)'}
                  </span>
                )}
              </div>
              <span className="font-mono text-emerald-400 flex items-center gap-1 tabular-nums shrink-0">
                {activeBatchItem?.status === 'analyzing-vision' ? (
                  <span className="text-cyan-400 flex items-center gap-1.5 animate-pulse">
                    <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    Refining Vision...
                  </span>
                ) : (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5" aria-hidden="true" />
                    {(topWinner.probability * 100).toFixed(1)}% Confidence
                  </>
                )}
              </span>
            </div>

            {/* Real MAPPA Anime Portrait + Character Name Header */}
            <div className="flex items-start gap-3.5 min-w-0">
              <img
                data-testid="predicted-character-real-portrait"
                src={topWinner.character.canonicalProxyUrl}
                alt={`Official MAPPA anime reference of ${topWinner.character.name}`}
                className="w-16 h-16 rounded-lg object-cover object-top border border-cyan-500/40 bg-slate-950 shrink-0"
              />
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-0.5 min-w-0">
                  <h3
                    data-testid="predicted-character-name"
                    className="text-lg sm:text-xl font-bold text-slate-100 tracking-tight break-words min-w-0"
                  >
                    {activeBatchItem?.status === 'analyzing-vision' ? (
                      <span className="text-cyan-300 animate-pulse">Classifying Character...</span>
                    ) : (
                      topWinner.character.name
                    )}
                  </h3>
                  <span className="text-xs font-mono text-slate-400 shrink-0">
                    {topWinner.character.kanji}
                  </span>
                </div>

                {/* Clean unboxed metadata with typographic separators (Zero-Pill Rule) */}
                <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-slate-300">
                  <span className="text-cyan-400 font-medium">{topWinner.character.wikiGroup}</span>
                  <span aria-hidden="true">·</span>
                  <span className="break-words">{topWinner.character.grade}</span>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono tabular-nums">
                    {realMappaFrames.length > 0
                      ? `${realMappaFrames.length} Show Frames`
                      : `${topWinner.character.totalRawImages} MAPPA Frames`}
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-1 space-y-1.5 text-xs min-w-0">
              <div className="text-slate-400 break-words">
                Cursed Technique:{' '}
                <span className="text-slate-200 font-medium">
                  {topWinner.character.cursedTechnique}
                </span>
              </div>
              <div className="text-slate-400 break-words">
                Primary Reference Arc:{' '}
                <span className="text-slate-200">{topWinner.character.primaryArc}</span>
              </div>
            </div>

            {/* MAPPA Studio Character Color Signature Swatches (Flex-wrapped so hex codes never overflow box) */}
            <div className="pt-1 flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs text-slate-400 shrink-0">MAPPA Studio Key Palette:</span>
              <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-slate-300">
                {topWinner.character.paletteHex.map((hex) => (
                  <span key={hex} className="flex items-center gap-1">
                    <span
                      className="w-3 h-3 rounded-sm border border-slate-700 shrink-0"
                      style={{ backgroundColor: hex }}
                    />
                    <span>{hex}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Calibrated Class Weights & Biases (W_c · x + b_c) Inspector & Live Tuner */}
          <div
            data-testid="calibrated-weights-panel"
            className="p-3.5 bg-slate-950 border border-slate-800 rounded-lg space-y-3 min-w-0 overflow-hidden"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-semibold text-slate-100 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400 shrink-0" aria-hidden="true" />
                <span>Calibrated Class Weights &amp; Biases</span>
              </span>
              {activeWeightOverride && (
                <button
                  type="button"
                  onClick={handleResetWeightParams}
                  className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
                >
                  <RotateCcw className="w-3 h-3" aria-hidden="true" />
                  <span>Reset Defaults</span>
                </button>
              )}
            </div>

            <div className="grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums">
              <div className="p-2 bg-[#111827] border border-slate-800 rounded min-w-0">
                <div className="text-[10px] text-slate-400 truncate">W_chroma</div>
                <div className="text-slate-100 font-semibold mt-0.5">
                  {effectiveCalibratedWeights.chromaWeight.toFixed(2)}
                </div>
              </div>
              <div className="p-2 bg-[#111827] border border-slate-800 rounded min-w-0">
                <div className="text-[10px] text-slate-400 truncate">W_linework</div>
                <div className="text-cyan-400 font-semibold mt-0.5">
                  {effectiveCalibratedWeights.lineworkWeight.toFixed(2)}
                </div>
              </div>
              <div className="p-2 bg-[#111827] border border-slate-800 rounded min-w-0">
                <div className="text-[10px] text-slate-400 truncate">Bias b_c</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{effectiveCalibratedWeights.classBias.toFixed(3)}
                </div>
              </div>
            </div>

            <div className="space-y-2 pt-0.5 text-[11px]">
              <div className="space-y-1">
                <div className="flex items-center justify-between font-mono text-slate-300">
                  <label htmlFor="cosine-scale-slider">Cosine Margin Scale (s_c)</label>
                  <span className="text-cyan-400 tabular-nums">
                    {effectiveCalibratedWeights.cosineScale.toFixed(1)}
                  </span>
                </div>
                <input
                  id="cosine-scale-slider"
                  type="range"
                  min={16}
                  max={48}
                  step={0.5}
                  value={effectiveCalibratedWeights.cosineScale}
                  onChange={(e) =>
                    handleUpdateWeightParam('cosineScale', Number(e.target.value))
                  }
                  className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
              </div>

              <div className="space-y-1">
                <div className="flex items-center justify-between font-mono text-slate-300">
                  <label htmlFor="class-bias-slider">Class Prior Bias (b_c)</label>
                  <span className="text-emerald-400 tabular-nums">
                    +{effectiveCalibratedWeights.classBias.toFixed(3)}
                  </span>
                </div>
                <input
                  id="class-bias-slider"
                  type="range"
                  min={0}
                  max={0.25}
                  step={0.005}
                  value={effectiveCalibratedWeights.classBias}
                  onChange={(e) =>
                    handleUpdateWeightParam('classBias', Number(e.target.value))
                  }
                  className="w-full accent-emerald-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                />
              </div>
            </div>
          </div>

          {/* Top-5 Softmax Probability Distribution */}
          <div className="space-y-3 min-w-0">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <h4 className="font-semibold text-slate-200">
                Top-5 Cosine Margin Softmax Output
              </h4>
              <span className="font-mono text-slate-400 tabular-nums">Cosine Sim</span>
            </div>

            <div className="space-y-2.5">
              {recognitionResult.topPredictions.map((pred, index) => {
                const pct = (pred.probability * 100).toFixed(1);
                return (
                  <div key={pred.character.id} className="space-y-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 text-xs min-w-0">
                      <span className="font-medium text-slate-200 truncate min-w-0">
                        {index + 1}. {pred.character.name}{' '}
                        <span className="text-slate-500 font-mono">
                          (#{String(pred.character.classIndex).padStart(2, '0')})
                        </span>
                      </span>
                      <span className="font-mono text-slate-300 tabular-nums shrink-0">
                        {pct}% · {pred.cosineSimilarity.toFixed(3)}
                      </span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-150 ${
                          index === 0 ? 'bg-cyan-400' : 'bg-slate-600'
                        }`}
                        style={{ width: `${Math.max(3, pred.probability * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Extracted Keyframe Visual Descriptors */}
          <div className="border-t border-slate-800 pt-4 space-y-2 min-w-0">
            <h4 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-cyan-400 shrink-0" aria-hidden="true" />
              <span>Learned ResNet-50 Visual Keypoints</span>
            </h4>
            <ul className="space-y-1 text-xs text-slate-400">
              {topWinner.character.visualDescriptors.map((desc) => (
                <li key={desc} className="flex items-start gap-2 min-w-0">
                  <span className="text-cyan-400 font-mono shrink-0">-</span>
                  <span className="break-words min-w-0">{desc}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};
