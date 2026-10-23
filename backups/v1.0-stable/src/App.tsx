/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState } from 'react';
import { Navbar, ActiveTab } from './components/Navbar';
import { BatchUploadedItem, RecognitionWorkbench } from './components/RecognitionWorkbench';
import { DatasetRegistry } from './components/DatasetRegistry';
import { ArchitectureInspector } from './components/ArchitectureInspector';
import { PyTorchCodeSuite } from './components/PyTorchCodeSuite';
import {
  JJK_MAPPA_ROSTER,
  JJKCharacterClass,
  PyTorchPipelineConfig,
  generatePyTorchCodeBundle,
} from './utils/jjkClassifierLogic';

const BATCH_STORAGE_KEY = 'jjk_vision_batch_queue_v2';
const ACTIVE_BATCH_ID_KEY = 'jjk_vision_active_batch_id_v2';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workbench');
  const [selectedCharacter, setSelectedCharacter] = useState<JJKCharacterClass>(
    JJK_MAPPA_ROSTER[0]
  );
  const [batchQueue, setBatchQueue] = useState<BatchUploadedItem[]>([]);
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null);

  // Restore any persisted batch images from sessionStorage on initial mount
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(BATCH_STORAGE_KEY);
      const savedActiveId = sessionStorage.getItem(ACTIVE_BATCH_ID_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Array<Omit<BatchUploadedItem, 'imageElement'>>;
      if (!Array.isArray(parsed) || parsed.length === 0) return;

      Promise.all(
        parsed.map(
          (entry) =>
            new Promise<BatchUploadedItem | null>((resolve) => {
              const img = new Image();
              img.crossOrigin = 'anonymous';
              img.onload = () => resolve({ ...entry, imageElement: img });
              img.onerror = () => resolve(null);
              img.src = entry.dataUrl;
            })
        )
      ).then((rehydrated) => {
        const valid = rehydrated.filter((x): x is BatchUploadedItem => Boolean(x));
        if (valid.length > 0) {
          setBatchQueue((prev) => (prev.length > 0 ? prev : valid));
          const target = valid.find((v) => v.id === savedActiveId) || valid[0];
          setActiveBatchId((prev) => prev || target.id);
          if (target.result?.topPredictions?.[0]?.character) {
            setSelectedCharacter(target.result.topPredictions[0].character);
          }
        }
      });
    } catch {
      // Ignore storage restoration errors
    }
  }, []);

  // Persist batchQueue metadata to sessionStorage whenever it changes so images survive tab switches & refreshes
  useEffect(() => {
    try {
      if (batchQueue.length === 0) {
        sessionStorage.removeItem(BATCH_STORAGE_KEY);
        sessionStorage.removeItem(ACTIVE_BATCH_ID_KEY);
        return;
      }
      const serializable = batchQueue.slice(0, 12).map((item) => ({
        id: item.id,
        fileName: item.fileName,
        dataUrl: item.dataUrl,
        result: item.result,
        characterFrames: item.characterFrames,
        status: item.status,
      }));
      sessionStorage.setItem(BATCH_STORAGE_KEY, JSON.stringify(serializable));
      if (activeBatchId) {
        sessionStorage.setItem(ACTIVE_BATCH_ID_KEY, activeBatchId);
      }
    } catch {
      // Ignore storage quota errors on very large batches
    }
  }, [batchQueue, activeBatchId]);

  const [pipelineConfig, setPipelineConfig] = useState<PyTorchPipelineConfig>({
    numClasses: JJK_MAPPA_ROSTER.length,
    minImagesPerCharacter: 200,
    inputResolution: 224,
    freezeStages: 2,
    learningRate: 0.0003,
    weightDecay: 0.01,
    labelSmoothing: 0.1,
    useMixupCutmix: true,
    useCosineClassifierHead: true,
    augmentationMultiplier: 4,
  });

  const handleDownloadPyTorchBundle = () => {
    const bundle = generatePyTorchCodeBundle(pipelineConfig);
    const combinedContent = Object.entries(bundle)
      .map(
        ([filename, content]) =>
          `# ==============================================================================\n# FILE: ${filename}\n# ==============================================================================\n\n${content}`
      )
      .join('\n\n');

    const blob = new Blob([combinedContent], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'jjk_resnet50_mappa_suite.py';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleSelectTab = (tab: ActiveTab) => {
    setActiveTab(tab);
    // If returning to workbench and we have uploaded images in batchQueue, ensure an uploaded image stays active by default
    if (tab === 'workbench' && batchQueue.length > 0 && !activeBatchId) {
      const first = batchQueue[0];
      setActiveBatchId(first.id);
      if (first.result?.topPredictions?.[0]?.character) {
        setSelectedCharacter(first.result.topPredictions[0].character);
      }
    }
    window.scrollTo({ top: 0, behavior: 'instant' });
  };

  const handleInspectCharacterFromDataset = (char: JJKCharacterClass) => {
    setSelectedCharacter(char);
    setActiveTab('workbench');
    window.scrollTo({ top: 0, behavior: 'instant' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0B0E17] text-slate-100">
      <Navbar
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        onExportBundle={handleDownloadPyTorchBundle}
      />

      <main className="flex-1 w-full max-w-[1380px] mx-auto px-4 sm:px-6 py-8">
        <div className={activeTab === 'workbench' ? 'block' : 'hidden'}>
          <RecognitionWorkbench
            selectedCharacter={selectedCharacter}
            onSelectCharacter={setSelectedCharacter}
            onNavigateTab={handleSelectTab}
            batchQueue={batchQueue}
            setBatchQueue={setBatchQueue}
            activeBatchId={activeBatchId}
            setActiveBatchId={setActiveBatchId}
            isActive={activeTab === 'workbench'}
          />
        </div>

        {activeTab === 'dataset' && (
          <DatasetRegistry
            onInspectCharacterInWorkbench={handleInspectCharacterFromDataset}
          />
        )}

        {activeTab === 'architecture' && <ArchitectureInspector />}

        {activeTab === 'code' && (
          <PyTorchCodeSuite
            config={pipelineConfig}
            onChangeConfig={setPipelineConfig}
            onDownloadAll={handleDownloadPyTorchBundle}
          />
        )}
      </main>

      <footer className="border-t border-slate-800/80 py-6 px-6 text-xs text-slate-500">
        <div className="max-w-[1380px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <span>
            JJK ResNet-50 Vision Studio · {JJK_MAPPA_ROSTER.length}-Class JJK Wiki MAPPA Art Style &amp; Fanart Recognition Pipeline
          </span>
          <div className="flex items-center gap-6">
            <button
              type="button"
              onClick={() => handleSelectTab('dataset')}
              className="hover:text-slate-300 transition-colors"
            >
              Dataset Manifest ({JJK_MAPPA_ROSTER.length} Wiki Classes · S1, S2, JJK 0 &amp; S3 Culling Game)
            </button>
            <button
              type="button"
              onClick={() => handleSelectTab('code')}
              className="hover:text-slate-300 transition-colors"
            >
              PyTorch ResNet-50 Source
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
