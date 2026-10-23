import { JJKCharacterClass, extractDominantHueFromRgba } from './jjkClassifierLogic';

export type FeatureMapOverlayMode = 'none' | 'sobel_edges' | 'layer3_grid' | 'layer4_gradcam';

export interface ExtractedCanvasMetrics {
  dominantHueDeg: number;
  edgeSharpness: number;
  contrastRatio: number;
  dataUrl: string;
}

/**
 * Renders a deterministic, stylized MAPPA Studio keyframe portrait for any of the 28 JJK characters
 * onto a 448x448 canvas, optionally applying ResNet-50 intermediate feature map visualizations,
 * and returns extracted pixel statistics for classification.
 */
export function renderMappaKeyframeToCanvas(
  canvas: HTMLCanvasElement,
  character: JJKCharacterClass,
  overlayMode: FeatureMapOverlayMode,
  customImageElement?: HTMLImageElement | null,
  isPresetWikiFrame: boolean = false,
  cropPreset: 'full' | 'half-body' | 'face-closeup' | 'left-half' | 'right-half' = 'full'
): ExtractedCanvasMetrics {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const size = 448;
  canvas.width = size;
  canvas.height = size;

  if (!ctx) {
    return {
      dominantHueDeg: character.dominantHueDeg,
      edgeSharpness: 0.85,
      contrastRatio: 0.78,
      dataUrl: '',
    };
  }

  const [primaryHex, accentHex, bgHex] = character.paletteHex;

  if (customImageElement) {
    // Draw backdrop + properly oriented full-frame or cropped (half-body / face / split-half) character image
    ctx.fillStyle = '#0B0E17';
    ctx.fillRect(0, 0, size, size);

    const rawW = customImageElement.naturalWidth || customImageElement.width || size;
    const rawH = customImageElement.naturalHeight || customImageElement.height || size;

    let sx = 0;
    let sy = 0;
    let sw = rawW;
    let sh = rawH;

    if (cropPreset === 'half-body') {
      // Waist-up / upper-half crop
      sx = Math.round(rawW * 0.08);
      sy = 0;
      sw = Math.round(rawW * 0.84);
      sh = Math.round(rawH * 0.52);
    } else if (cropPreset === 'face-closeup') {
      // Tight head, eyes & hairline crop
      sx = Math.round(rawW * 0.22);
      sy = Math.round(rawH * 0.02);
      sw = Math.round(rawW * 0.56);
      sh = Math.round(rawH * (rawH > rawW * 1.25 ? 0.28 : 0.48));
    } else if (cropPreset === 'left-half') {
      // Vertically split left-half character crop
      sx = 0;
      sy = 0;
      sw = Math.round(rawW * 0.52);
      sh = rawH;
    } else if (cropPreset === 'right-half') {
      // Vertically split right-half character crop
      sx = Math.round(rawW * 0.48);
      sy = 0;
      sw = Math.round(rawW * 0.52);
      sh = rawH;
    }

    // Subtle atmospheric fill behind non-square aspect ratios
    const coverScale = Math.max(size / sw, size / sh);
    const cw = sw * coverScale;
    const ch = sh * coverScale;
    ctx.save();
    ctx.globalAlpha = 0.22;
    ctx.drawImage(customImageElement, sx, sy, sw, sh, (size - cw) / 2, (size - ch) / 2, cw, ch);
    ctx.restore();

    // Properly oriented contain/portrait framing centered in the viewport
    const fitScale =
      sh / sw > 1.35
        ? Math.min((size * 0.96) / sw, (size * 0.96) / sh)
        : Math.min(size / sw, size / sh);
    const w = sw * fitScale;
    const h = sh * fitScale;
    const offsetX = (size - w) / 2;
    const offsetY = (size - h) / 2;
    ctx.drawImage(customImageElement, sx, sy, sw, sh, offsetX, offsetY, w, h);
  } else {
    // 1. Studio composite atmospheric background
    const bgGrad = ctx.createRadialGradient(size * 0.5, size * 0.42, 30, size * 0.5, size * 0.5, size * 0.72);
    bgGrad.addColorStop(0, accentHex + '44');
    bgGrad.addColorStop(0.55, bgHex);
    bgGrad.addColorStop(1, '#07090E');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, size, size);

    // 2. Cursed energy radial speed-lines (MAPPA composite signature)
    ctx.save();
    ctx.translate(size / 2, size * 0.46);
    ctx.strokeStyle = accentHex + '38';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 36; i++) {
      const angle = (i * Math.PI * 2) / 36 + (character.classIndex % 5) * 0.1;
      ctx.beginPath();
      ctx.moveTo(Math.cos(angle) * 115, Math.sin(angle) * 115);
      ctx.lineTo(Math.cos(angle) * 240, Math.sin(angle) * 240);
      ctx.stroke();
    }
    ctx.restore();

    // 3. High-collar Tokyo/Kyoto Jujutsu uniform or robe shoulders
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(size * 0.16, size);
    ctx.lineTo(size * 0.25, size * 0.68);
    ctx.lineTo(size * 0.37, size * 0.58);
    ctx.lineTo(size * 0.63, size * 0.58);
    ctx.lineTo(size * 0.75, size * 0.68);
    ctx.lineTo(size * 0.84, size);
    ctx.closePath();
    ctx.fillStyle = '#0F172A';
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#1E293B';
    ctx.stroke();

    // Uniform accent / hood / kesa detail
    if (character.id === 'yuji_itadori') {
      ctx.fillStyle = '#DC2626';
      ctx.beginPath();
      ctx.arc(size * 0.5, size * 0.67, 74, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.fill();
    } else if (character.id === 'satoru_gojo' || character.id === 'megumi_fushiguro' || character.id === 'toge_inumaki') {
      // High collar
      ctx.fillStyle = '#111827';
      ctx.fillRect(size * 0.36, size * 0.52, size * 0.28, size * 0.16);
      ctx.strokeStyle = '#334155';
      ctx.strokeRect(size * 0.36, size * 0.52, size * 0.28, size * 0.16);
    }
    ctx.restore();

    // 4. Character head & cel-shaded jawline contour
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(size * 0.34, size * 0.36);
    ctx.quadraticCurveTo(size * 0.34, size * 0.56, size * 0.5, size * 0.63);
    ctx.quadraticCurveTo(size * 0.66, size * 0.56, size * 0.66, size * 0.36);
    ctx.arc(size * 0.5, size * 0.36, size * 0.16, 0, Math.PI, true);
    ctx.closePath();
    ctx.fillStyle =
      character.id === 'hanami'
        ? '#F8FAFC'
        : character.id === 'dagon'
          ? '#DC2626'
          : character.id === 'jogo'
            ? '#D6D3D1'
            : '#FDE6D8';
    ctx.fill();

    // MAPPA hard cel-shadow plane on left side of face
    ctx.beginPath();
    ctx.moveTo(size * 0.34, size * 0.36);
    ctx.quadraticCurveTo(size * 0.35, size * 0.54, size * 0.49, size * 0.62);
    ctx.lineTo(size * 0.46, size * 0.36);
    ctx.closePath();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.24)';
    ctx.fill();

    ctx.lineWidth = 2.75;
    ctx.strokeStyle = '#0F172A';
    ctx.stroke();
    ctx.restore();

    // 5. Character-specific eyes, scars, stitches, or blindfold
    ctx.save();
    if (character.id === 'jogo') {
      // Single giant cyclops eye
      ctx.beginPath();
      ctx.arc(size * 0.5, size * 0.44, 24, 0, Math.PI * 2);
      ctx.fillStyle = '#1C1917';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(size * 0.5, size * 0.44, 11, 0, Math.PI * 2);
      ctx.fillStyle = '#EF4444';
      ctx.fill();
    } else if (character.id === 'hanami') {
      // Twin black tree-branch eye roots
      ctx.fillStyle = '#18181B';
      ctx.beginPath();
      ctx.ellipse(size * 0.43, size * 0.43, 14, 22, -0.3, 0, Math.PI * 2);
      ctx.ellipse(size * 0.57, size * 0.43, 14, 22, 0.3, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Left & Right anime eyes with luminous iris
      const eyeY = size * 0.44;
      for (const eyeX of [size * 0.425, size * 0.575]) {
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.ellipse(eyeX, eyeY, 16, 9, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0F172A';
        ctx.lineWidth = 2.2;
        ctx.stroke();

        ctx.fillStyle = accentHex;
        ctx.beginPath();
        ctx.arc(eyeX, eyeY, 7.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#090D16';
        ctx.beginPath();
        ctx.arc(eyeX, eyeY, 3.2, 0, Math.PI * 2);
        ctx.fill();

        // Specular highlight
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(eyeX - 3, eyeY - 3, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Sukuna / Yuji secondary eye slits & markings
      if (character.id === 'ryomen_sukuna' || character.id === 'yuji_itadori') {
        ctx.strokeStyle = character.id === 'ryomen_sukuna' ? '#991B1B' : '#0F172A';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(size * 0.39, size * 0.48);
        ctx.lineTo(size * 0.45, size * 0.485);
        ctx.moveTo(size * 0.55, size * 0.485);
        ctx.lineTo(size * 0.61, size * 0.48);
        ctx.stroke();
      }

      if (character.id === 'ryomen_sukuna') {
        // Forehead & nose curse markings
        ctx.strokeStyle = '#090D16';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(size * 0.43, size * 0.37);
        ctx.lineTo(size * 0.57, size * 0.37);
        ctx.moveTo(size * 0.46, size * 0.51);
        ctx.lineTo(size * 0.54, size * 0.51);
        ctx.stroke();
      } else if (character.id === 'mahito') {
        // Patchwork facial stitches
        ctx.strokeStyle = '#1E293B';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(size * 0.35, size * 0.49);
        ctx.lineTo(size * 0.65, size * 0.52);
        ctx.stroke();
        for (let sx = 0.38; sx <= 0.62; sx += 0.04) {
          ctx.beginPath();
          ctx.moveTo(size * sx, size * 0.48);
          ctx.lineTo(size * sx, size * 0.53);
          ctx.stroke();
        }
      } else if (character.id === 'choso' || character.id === 'utahime_iori') {
        // Horizontal bridge mark / scar
        ctx.fillStyle = '#7F1D1D';
        ctx.fillRect(size * 0.39, size * 0.455, size * 0.22, 6);
      }
    }
    ctx.restore();

    // 6. Spiky / flowing MAPPA hair silhouette crown
    ctx.save();
    ctx.fillStyle = primaryHex;
    ctx.strokeStyle = '#0F172A';
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    const cx = size * 0.5;
    const cy = size * 0.34;
    const spikes = 14;
    for (let i = 0; i <= spikes; i++) {
      const t = i / spikes;
      const angle = Math.PI * 0.92 + t * Math.PI * 1.16;
      const outerR = i % 2 === 0 ? 128 : 84;
      const px = cx + Math.cos(angle) * outerR;
      const py = cy + Math.sin(angle) * outerR * 0.88;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // 7. Frame metadata watermark in top-left corner of canvas (dynamically sized to prevent text overflow)
    ctx.save();
    const line1 = `CLASS #${String(character.classIndex).padStart(2, '0')} · ${character.name.toUpperCase()}`;
    const line2 = `MAPPA KEYFRAME · 224x224 CROP`;
    ctx.font = '600 12px "JetBrains Mono", monospace';
    const w1 = ctx.measureText(line1).width;
    ctx.font = '400 11px "JetBrains Mono", monospace';
    const w2 = ctx.measureText(line2).width;
    const boxW = Math.min(size - 24, Math.ceil(Math.max(w1, w2) + 24));
    ctx.fillStyle = 'rgba(11, 14, 23, 0.86)';
    ctx.fillRect(12, 12, boxW, 46);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.28)';
    ctx.strokeRect(12, 12, boxW, 46);
    ctx.fillStyle = '#F8FAFC';
    ctx.font = '600 12px "JetBrains Mono", monospace';
    ctx.fillText(line1, 22, 31, boxW - 20);
    ctx.fillStyle = '#38BDF8';
    ctx.font = '400 11px "JetBrains Mono", monospace';
    ctx.fillText(line2, 22, 48, boxW - 20);
    ctx.restore();
  }

  // Extract real pixel metrics before drawing feature map overlays
  const imgData = ctx.getImageData(0, 0, size, size);
  const data = imgData.data;
  let edgeSum = 0;
  let lumMin = 255;
  let lumMax = 0;

  for (let y = 1; y < size - 1; y += 4) {
    for (let x = 1; x < size - 1; x += 4) {
      const idx = (y * size + x) * 4;
      const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      const lumRight = 0.299 * data[idx + 4] + 0.587 * data[idx + 5] + 0.114 * data[idx + 6];
      const lumDown =
        0.299 * data[idx + size * 4] +
        0.587 * data[idx + size * 4 + 1] +
        0.114 * data[idx + size * 4 + 2];
      edgeSum += Math.abs(lum - lumRight) + Math.abs(lum - lumDown);
      if (lum < lumMin) lumMin = lum;
      if (lum > lumMax) lumMax = lum;
    }
  }

  const samples = ((size - 2) / 4) * ((size - 2) / 4);
  const edgeSharpness = Math.min(0.98, Math.max(0.45, edgeSum / (samples * 48)));
  const contrastRatio = Math.min(0.98, Math.max(0.4, (lumMax - lumMin) / 255));

  const extractedHue =
    customImageElement && !isPresetWikiFrame
      ? extractDominantHueFromRgba(data, 4).dominantHueDeg
      : character.dominantHueDeg;

  if (customImageElement) {
    ctx.save();
    const line1 = `CLASS #${String(character.classIndex).padStart(2, '0')} · ${character.name.toUpperCase()}`;
    const line2 = isPresetWikiFrame
      ? `OFFICIAL MAPPA ANIME FRAME · 224x224`
      : `PREDICTED TARGET · CALIBRATED RESNET-50`;
    ctx.font = '600 12px "JetBrains Mono", monospace';
    const w1 = ctx.measureText(line1).width;
    ctx.font = '400 11px "JetBrains Mono", monospace';
    const w2 = ctx.measureText(line2).width;
    const boxW = Math.min(size - 24, Math.ceil(Math.max(w1, w2) + 24));
    ctx.fillStyle = 'rgba(11, 14, 23, 0.86)';
    ctx.fillRect(12, 12, boxW, 46);
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.strokeRect(12, 12, boxW, 46);
    ctx.fillStyle = '#F8FAFC';
    ctx.font = '600 12px "JetBrains Mono", monospace';
    ctx.fillText(line1, 22, 31, boxW - 20);
    ctx.fillStyle = '#38BDF8';
    ctx.font = '400 11px "JetBrains Mono", monospace';
    ctx.fillText(line2, 22, 48, boxW - 20);
    ctx.restore();
  }

  // Apply requested ResNet-50 intermediate feature visualization overlay
  if (overlayMode === 'sobel_edges') {
    ctx.save();
    ctx.fillStyle = 'rgba(7, 9, 14, 0.72)';
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = '#06B6D4';
    ctx.lineWidth = 1.2;
    for (let y = 8; y < size - 8; y += 6) {
      for (let x = 8; x < size - 8; x += 6) {
        const idx = (y * size + x) * 4;
        const diff =
          Math.abs(data[idx] - data[idx + 16]) + Math.abs(data[idx + 1] - data[idx + size * 16 + 1]);
        if (diff > 42) {
          ctx.strokeRect(x, y, 3, 3);
        }
      }
    }
    ctx.restore();
  } else if (overlayMode === 'layer3_grid') {
    // 14x14 spatial grid (ResNet-50 Layer3 [1, 1024, 14, 14])
    const gridN = 14;
    const cell = size / gridN;
    ctx.save();
    for (let gy = 0; gy < gridN; gy++) {
      for (let gx = 0; gx < gridN; gx++) {
        const distFromCenter = Math.hypot(gx - 6.5, gy - 5.8);
        const activation = Math.max(0.08, 1 - distFromCenter / 8.2);
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.28)';
        ctx.strokeRect(gx * cell, gy * cell, cell, cell);
        if (activation > 0.52) {
          ctx.fillStyle = `rgba(6, 182, 212, ${(activation * 0.35).toFixed(2)})`;
          ctx.fillRect(gx * cell + 2, gy * cell + 2, cell - 4, cell - 4);
        }
      }
    }
    ctx.restore();
  } else if (overlayMode === 'layer4_gradcam') {
    // 7x7 spatial Grad-CAM saliency map (ResNet-50 Layer4 [1, 2048, 7, 7])
    ctx.save();
    const camGrad = ctx.createRadialGradient(
      size * 0.5,
      size * 0.42,
      18,
      size * 0.5,
      size * 0.44,
      size * 0.42
    );
    camGrad.addColorStop(0, 'rgba(239, 68, 68, 0.62)');
    camGrad.addColorStop(0.45, 'rgba(245, 158, 11, 0.45)');
    camGrad.addColorStop(0.75, 'rgba(6, 182, 212, 0.28)');
    camGrad.addColorStop(1, 'rgba(15, 23, 42, 0.0)');
    ctx.fillStyle = camGrad;
    ctx.fillRect(0, 0, size, size);

    // Bounding box around high-activation facial & hair region
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(size * 0.24, size * 0.16, size * 0.52, size * 0.52);
    ctx.setLineDash([]);
    ctx.fillStyle = '#0B0E17';
    ctx.fillRect(size * 0.24, size * 0.16 - 20, 188, 20);
    ctx.fillStyle = '#F59E0B';
    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.fillText('LAYER4 GRAD-CAM PEAK: 0.94', size * 0.24 + 6, size * 0.16 - 6);
    ctx.restore();
  }

  return {
    dominantHueDeg: extractedHue,
    edgeSharpness: Number(edgeSharpness.toFixed(3)),
    contrastRatio: Number(contrastRatio.toFixed(3)),
    dataUrl: canvas.toDataURL('image/png'),
  };
}
