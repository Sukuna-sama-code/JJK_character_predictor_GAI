import React, { useState } from 'react';
import {
  PyTorchPipelineConfig,
  generatePyTorchCodeBundle,
} from '../utils/jjkClassifierLogic';
import { Copy, Check, Download, Code2 } from 'lucide-react';

interface PyTorchCodeSuiteProps {
  config: PyTorchPipelineConfig;
  onChangeConfig: (next: PyTorchPipelineConfig) => void;
  onDownloadAll: () => void;
}

export const PyTorchCodeSuite: React.FC<PyTorchCodeSuiteProps> = ({
  config,
  onChangeConfig,
  onDownloadAll,
}) => {
  const codeBundle = generatePyTorchCodeBundle(config);
  const fileNames = Object.keys(codeBundle);
  const [activeFile, setActiveFile] = useState<string>(fileNames[0]);
  const [copied, setCopied] = useState<boolean>(false);

  const currentCode = codeBundle[activeFile] || codeBundle[fileNames[0]];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Fallback copy state
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <section aria-labelledby="code-heading" className="space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-6 border-b border-slate-800">
        <div className="space-y-2 max-w-3xl">
          <p className="text-xs font-mono text-cyan-400 tracking-wide">
            PRODUCTION PYTORCH CODE GENERATOR · RESNET-50 + MAPPA SCRAPER PIPELINE
          </p>
          <h1
            id="code-heading"
            className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100"
          >
            PyTorch ResNet-50 Vision Model & MAPPA Dataset Curation Scripts
          </h1>
          <p className="text-sm text-slate-400 leading-relaxed">
            Configure your ResNet-50 backbone parameters below to dynamically generate the complete
            Python training repository: the ResNet-50 dual-pooling architecture with Cosine Margin
            head, the official MAPPA Studio dataset curator enforcing ≥{config.minImagesPerCharacter}{' '}
            images per character with 64-bit perceptual hash deduplication, and the mixed-precision
            AdamW training loop.
          </p>
        </div>

        <button
          type="button"
          onClick={onDownloadAll}
          className="px-4 py-2.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors flex items-center gap-2 whitespace-nowrap shrink-0"
        >
          <Download className="w-4 h-4" aria-hidden="true" />
          <span>Download Active Script ({activeFile})</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Hyperparameter Configuration Column (4 cols) */}
        <div className="lg:col-span-4 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-5">
          <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Code2 className="w-4 h-4 text-cyan-400" aria-hidden="true" />
            Pipeline Hyperparameters
          </h2>

          <div className="space-y-4 text-xs">
            <div>
              <label
                htmlFor="input-res-select"
                className="block text-slate-400 font-medium mb-1.5"
              >
                Input Tensor Resolution
              </label>
              <select
                id="input-res-select"
                value={config.inputResolution}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    inputResolution: Number(e.target.value) as 224 | 288 | 384,
                  })
                }
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-100 font-mono"
              >
                <option value={224}>224×224 (Standard ResNet-50 · 7×7 Layer4)</option>
                <option value={288}>288×288 (High-Detail · 9×9 Layer4)</option>
                <option value={384}>384×384 (Fine Linework · 12×12 Layer4)</option>
              </select>
            </div>

            <div>
              <label
                htmlFor="min-images-input"
                className="flex items-center justify-between text-slate-400 font-medium mb-1.5"
              >
                <span>Minimum MAPPA Images / Character</span>
                <span className="font-mono text-cyan-400 tabular-nums">
                  ≥ {config.minImagesPerCharacter} imgs
                </span>
              </label>
              <input
                id="min-images-input"
                type="number"
                min={200}
                max={500}
                step={10}
                value={config.minImagesPerCharacter}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    minImagesPerCharacter: Math.max(200, Number(e.target.value) || 200),
                  })
                }
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-100 font-mono tabular-nums"
              />
            </div>

            <div>
              <label
                htmlFor="code-freeze-select"
                className="block text-slate-400 font-medium mb-1.5"
              >
                Frozen ResNet-50 Stages (IMAGENET1K_V2)
              </label>
              <select
                id="code-freeze-select"
                value={config.freezeStages}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    freezeStages: Number(e.target.value) as 0 | 1 | 2 | 3,
                  })
                }
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-100"
              >
                <option value={0}>0 — Unfreeze Entire ResNet-50</option>
                <option value={1}>1 — Freeze Stem + Layer1</option>
                <option value={2}>2 — Freeze Stem + Layer1 + Layer2 (Recommended)</option>
                <option value={3}>3 — Freeze Through Layer3</option>
              </select>
            </div>

            <div className="pt-2 flex items-center justify-between">
              <label htmlFor="cosine-head-toggle" className="text-slate-300 font-medium">
                Use Cosine Margin Classifier Head
              </label>
              <input
                id="cosine-head-toggle"
                type="checkbox"
                checked={config.useCosineClassifierHead}
                onChange={(e) =>
                  onChangeConfig({
                    ...config,
                    useCosineClassifierHead: e.target.checked,
                  })
                }
                className="w-4 h-4 accent-cyan-400 rounded cursor-pointer"
              />
            </div>
          </div>

          <div className="border-t border-slate-800 pt-4 space-y-2 text-xs text-slate-400">
            <div className="font-semibold text-slate-200">Quick Environment Setup</div>
            <pre className="p-3 bg-slate-950 border border-slate-800 rounded-lg font-mono text-[11px] text-cyan-300 overflow-x-auto">
              pip install torch torchvision pillow
            </pre>
          </div>
        </div>

        {/* Right Code Viewer Column (8 cols) */}
        <div className="lg:col-span-8 bg-[#111827] border border-slate-800 rounded-xl overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 bg-slate-950 border-b border-slate-800">
            <div className="flex items-center gap-1.5 overflow-x-auto">
              {fileNames.map((fname) => (
                <button
                  key={fname}
                  type="button"
                  onClick={() => setActiveFile(fname)}
                  className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors whitespace-nowrap ${
                    activeFile === fname
                      ? 'bg-cyan-400 text-slate-950 font-semibold'
                      : 'text-slate-400 hover:text-slate-200 bg-slate-900'
                  }`}
                >
                  {fname}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={handleCopy}
              className="px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-900 border border-slate-700 rounded-md hover:bg-slate-800 transition-colors flex items-center gap-1.5 whitespace-nowrap"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" aria-hidden="true" />
                  <span>Copied Code</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-cyan-400" aria-hidden="true" />
                  <span>Copy {activeFile}</span>
                </>
              )}
            </button>
          </div>

          <pre
            data-testid="pytorch-code-viewer"
            className="p-5 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed max-h-[620px] overflow-y-auto"
          >
            <code>{currentCode}</code>
          </pre>
        </div>
      </div>
    </section>
  );
};
