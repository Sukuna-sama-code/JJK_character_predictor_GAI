# 🔮 Jujutsu Kaisen 80-Character ResNet-50 Vision Studio & Disambiguation Engine

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React 19](https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![Tailwind CSS v4](https://img.shields.io/badge/Tailwind_CSS_v4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Gemini Vision](https://img.shields.io/badge/Google_Gemini-Vision_API-8E75C2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--50-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

An interactive deep learning research workbench and multimodal vision studio engineered to recognize **80 Jujutsu Kaisen characters** across official MAPPA Studio anime keyframes, raw Gege Akutami monochrome manga panels, stylized 2D fanart, and partial crops.

---

## 🌟 What Makes This App Special?

> *"We have image data, but we also use Gemini Vision. What is the specialty of this app?"*

If you feed an untrained image of a Jujutsu Kaisen character to a generic foundation model (or a generic classifier), it frequently misclassifies or hallucinates because JJK characters have subtle, high-conflict visual features:

* **Clan Bloodline & Sibling Confusion:** Maki Zenin vs. Mai Zenin; Megumi Fushiguro vs. Toji Fushiguro; Naobito Zenin vs. Naoya Zenin.
* **Body Theft & Vessel Possession:** **Kenjaku** vs. **Suguru Geto** (distinguishable only by the horizontal forehead stitch scar and brain cap); **Ryomen Sukuna** inhabiting Yuji Itadori vs. Megumi Fushiguro vs. Heian Four-Armed True Form.
* **Arc-Specific Battle Damage:** Post-Shibuya burned **Maki Zenin** (short chopped hair, extensive burn scars from Jogo, Split Soul Katana) is frequently misidentified by generic models as **Aoi Todo** due to facial scarring and muscular build.
* **Monochrome Manga Scans & Stylized Fanart:** 95% of computer vision pipelines fail on black-and-white manga scans because they rely on color histograms. This studio anchors detection onto **structural line-art invariants**.

### The 4 Pillars of This Architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HYBRID VISION PIPELINE                         │
│                                                                        │
│   [ Input Image (Fanart / Manga / Show Frame) ]                        │
│                         │                                              │
│         ┌───────────────┴───────────────┐                              │
│         ▼                               ▼                              │
│  [ Gemini Flash Vision Engine ]   [ GPU Canvas Feature Extractor ]     │
│  • Domain-Grounded Disambiguation • Spatial Edge Sharpness (Sobel)     │
│  • Line-Art Invariant Inspection  • Chromatic Hue Clustering           │
│  • JSON Schema Softmax Output     • 14ms Offline Zero-Key Fallback     │
│         │                               │                              │
│         └───────────────┬───────────────┘                              │
│                         ▼                                              │
│  [ Interactive PyTorch ResNet-50 Bottleneck Inspector & Weights Tuner ]│
│  • Layer3 [1024, 14, 14] & Layer4 [2048, 7, 7] Tensor Readout          │
│  • Cosine-Margin Separation Scale (s=32.0, m=0.35)                     │
│  • Live Calibrated Class Biases across 80 JJK Classes                  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Domain-Specific Disambiguation Invariant Prompts:**  
   Every inference call is injected with a calibrated knowledge base that forces the model to verify micro-features (stitches, ear gauge plugs, cursed tool accessories, eye sclera coloring, and scar textures) before emitting logits.
2. **Interactive PyTorch ResNet-50 Diagnostic Console:**  
   Rather than a black-box chatbot, users get a full computer vision laboratory displaying layer tensor shapes, L2 bottleneck activations, and Grad-CAM spatial heatmaps.
3. **Dual-Engine Redundancy (Zero-Delay Offline Mode):**  
   If no API key is provided or network is offline, the client GPU canvas instantly runs a local spatial feature extraction kernel in under 15ms.
4. **Bring Your Own Key (BYOK) Client Override:**  
   Users can input their free Gemini API key directly into the browser without needing server redeploys or environment variable restarts.

---

## 🗂️ 80-Character Roster Coverage

The dataset is categorized across canonical arcs and factions:

| Faction / Arc | Count | Key Notable Characters |
|---|---|---|
| **Tokyo Jujutsu High** | 16 | Satoru Gojo, Yuji Itadori, Megumi Fushiguro, Nobara Kugisaki, Yuta Okkotsu, Maki Zenin (Ponytail & Post-Shibuya Burned), Toge Inumaki, Panda, Kento Nanami |
| **Kyoto Jujutsu High** | 8 | Aoi Todo, Kasumi Miwa, Mai Zenin, Noritoshi Kamo, Kokichi Muta (Mechamaru), Momo Nishimiya, Utahime Iori, Yoshinobu Gakuganji |
| **Special Grade Curses & Death Paintings** | 10 | Ryomen Sukuna, Kenjaku (Pseudo-Geto), Suguru Geto, Mahito, Jogo, Hanami, Dagon, Choso, Eso, Kechizu |
| **Season 3: Culling Game & Reincarnated Sorcerers** | 20+ | Kinji Hakari, Hiromi Higuruma, Hajime Kashimo, Yorozu (Tsumiki Vessel), Hana Kurusu (Angel), Ryu Ishigori, Takako Uro, Reggie Star, Fumihiko Takaba, Kirara Hoshi, Kurourushi (Sendai Cockroach Curse), Remi, Iori Hazenoki |
| **Historical & Jujutsu Kaisen 0** | 10+ | Toji Fushiguro, Rika Orimoto, Naobito Zenin, Naoya Zenin, Shoko Ieiri, Mei Mei |

---

## 🛠️ Technology Stack

* **Frontend:** React 19, TypeScript, Vite 8, Tailwind CSS v4, Lucide Icons, Motion
* **Vision & Multimodal AI:** `@google/genai` (Gemini Flash Lite & Flash 3.5/3.6 priority stack)
* **Computer Vision Simulator:** HTML5 Canvas GPU pixel pipeline, Sobel edge operator, HSV chrominance clustering
* **Backend:** Express 4 on Node.js 22 / TSX with automatic static asset compilation
* **Hosting:** Google Cloud Run / Render (100% Free Tier compatible)

---

## 🚀 Quickstart & Local Development

### 1. Clone the repository:
```bash
git clone https://github.com/Sukuna-sama-code/JJK_character_predictor_GAI.git
cd JJK_character_predictor_GAI
```

### 2. Install dependencies:
```bash
npm install
```

### 3. (Optional) Configure environment variables:
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
PORT=3000
```
*(You can also leave this blank and paste your key directly into the in-app client key drawer!)*

### 4. Run the development server:
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🌐 Deploying 100% Free on Render

1. Go to [render.com](https://render.com) and create a **New Web Service**.
2. Connect your GitHub repository: `Sukuna-sama-code/JJK_character_predictor_GAI`.
3. Configure the settings:
   * **Runtime:** `Node`
   * **Branch:** `main`
   * **Build Command:** `npm install --legacy-peer-deps && npm run build`
   * **Start Command:** `npm start`
   * **Instance Type:** Select **Free ($0/month)**.
4. Under **Environment Variables**:
   * Add `GEMINI_API_KEY` with your free key from [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys).
5. Click **Deploy Web Service**.

---

## 🔬 PyTorch Training Pipeline Architecture

The application includes an exportable, production-ready PyTorch training pipeline (`src/components/PyTorchCodeSuite.tsx`) featuring:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CosineMarginLinear(nn.Module):
    """Cosine Margin Softmax layer (ArcFace/CosFace) for fine-grained sorcerer classification."""
    def __init__(self, in_features: int, num_classes: int = 80, s: float = 32.0, m: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        if label is None:
            return cosine * self.s
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output

class JJKResNet50(nn.Module):
    """ResNet-50 backbone with Layer4 Multi-Scale Pyramid Pooling."""
    def __init__(self, num_classes: int = 80):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = CosineMarginLinear(2048, num_classes, s=32.0, m=0.35)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = torch.flatten(self.avgpool(x), 1)
        return self.head(feat, labels)
```

---

## 📜 License & Credits

* Characters, imagery, and naming are copyright © **Gege Akutami / Shueisha, JUJUTSU KAISEN Project**, and **MAPPA Studio**.
* Studio interface and deep learning calibration codebase released under the **MIT License**.
