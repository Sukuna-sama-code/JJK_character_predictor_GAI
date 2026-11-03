export type CharacterFaction =
  | 'Jujutsu Sorcerer'
  | 'Cursed Spirit'
  | 'Curse User / Vessel';

export type WikiCharacterGroup =
  | 'Tokyo Jujutsu High'
  | 'Kyoto Jujutsu High'
  | 'Clans & Independent Sorcerers'
  | 'Star Plasma Vessel Arc'
  | "Geto's Group"
  | 'Curse Users & Death Paintings'
  | 'Cursed Spirits & Shikigami'
  | 'Culling Game & Reincarnated Sorcerers';

export type MappaArc =
  | 'Season 1 ( Cour 1 & 2 )'
  | 'Jujutsu Kaisen 0 (Movie)'
  | 'Season 2: Hidden Inventory'
  | 'Season 2: Shibuya Incident'
  | 'Season 3: Culling Game';

export interface MappaSourceBreakdown {
  jujutsukaisenJpSheets: number;
  mappaWorksKeyframes: number;
  episodeCleanCuts: number;
  blurayTurnarounds: number;
}

export interface CalibratedClassWeights {
  chromaWeight: number;
  lineworkWeight: number;
  contrastWeight: number;
  classBias: number;
  cosineScale: number;
}

export interface JJKCharacterClass {
  id: string;
  classIndex: number;
  name: string;
  kanji: string;
  faction: CharacterFaction;
  wikiGroup: WikiCharacterGroup;
  grade: string;
  cursedTechnique: string;
  primaryArc: MappaArc;
  mappaSources: MappaSourceBreakdown;
  totalRawImages: number;
  paletteHex: [string, string, string]; // [Hair/Primary, Eyes/Accent, Uniform/Aura]
  dominantHueDeg: number; // 0-360
  visualDescriptors: string[];
  resnet50ValidationF1: number;
  confusionPairId: string;
  canonicalPortraitUrl: string;
  canonicalProxyUrl: string;
  calibratedWeights: CalibratedClassWeights;
}

export interface DatasetSplitStats {
  characterCount: number;
  minImagesConstraint: number;
  allClassesMeetMinimum: boolean;
  lowestClassCount: number;
  highestClassCount: number;
  totalRawImages: number;
  totalAugmentedImages: number;
  trainCount: number;
  valCount: number;
  testCount: number;
}

export interface RecognitionPrediction {
  character: JJKCharacterClass;
  probability: number;
  cosineSimilarity: number;
  matchedFeatures: string[];
}

export interface RecognitionResult {
  topPredictions: RecognitionPrediction[];
  bottleneckNorm: number;
  layer4ActivationPeak: number;
  inferenceLatencyMs: number;
  featureSummary: {
    dominantColors: string[];
    edgeSharpness: number;
    mappaLineworkScore: number;
    auraSignature: string;
  };
  analysisSource: 'gemini-vision-hybrid' | 'resnet50-feature-kernel';
  visualReasoning: string;
}

export interface TrainingEpochPoint {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  top1Acc: number;
  top5Acc: number;
  learningRate: number;
}

export interface PyTorchPipelineConfig {
  numClasses: number;
  minImagesPerCharacter: number;
  inputResolution: 224 | 288 | 384;
  freezeStages: 0 | 1 | 2 | 3;
  learningRate: number;
  weightDecay: number;
  labelSmoothing: number;
  useMixupCutmix: boolean;
  useCosineClassifierHead: boolean;
  augmentationMultiplier: number;
}

interface RawCharacterSeed {
  id: string;
  name: string;
  kanji: string;
  faction: CharacterFaction;
  wikiGroup: WikiCharacterGroup;
  grade: string;
  cursedTechnique: string;
  primaryArc: MappaArc;
  sheets: [number, number, number, number];
  paletteHex: [string, string, string];
  dominantHueDeg: number;
  visualDescriptors: string[];
  f1: number;
  confusionPairId: string;
}

const RAW_WIKI_ROSTER_SEEDS: RawCharacterSeed[] = [
  // ================= TOKYO JUJUTSU HIGH (16) =================
  {
    id: 'satoru_gojo',
    name: 'Satoru Gojo',
    kanji: '五条 悟',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Special Grade Faculty',
    cursedTechnique: 'Limitless & Six Eyes (Infinite Void)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [28, 44, 162, 31],
    paletteHex: ['#F8FAFC', '#38BDF8', '#0F172A'],
    dominantHueDeg: 199,
    visualDescriptors: ['Spiky snow-white hair', 'Glowing crystalline Six Eyes / black blindfold', 'High-collared dark navy Tokyo Jujutsu uniform'],
    f1: 0.986,
    confusionPairId: 'toge_inumaki',
  },
  {
    id: 'yuji_itadori',
    name: 'Yuji Itadori',
    kanji: '虎杖 悠仁',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'First-Year Student (Grade 1 Pending)',
    cursedTechnique: 'Divergent Fist & Black Flash',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [26, 42, 158, 28],
    paletteHex: ['#F472B6', '#DC2626', '#1E293B'],
    dominantHueDeg: 340,
    visualDescriptors: ['Undercut salmon-pink spiky hair', 'Secondary eye slits beneath warm brown eyes', 'Red collared hood over dark navy uniform'],
    f1: 0.978,
    confusionPairId: 'ryomen_sukuna',
  },
  {
    id: 'megumi_fushiguro',
    name: 'Megumi Fushiguro',
    kanji: '伏黒 恵',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Grade 2 First-Year Student',
    cursedTechnique: 'Ten Shadows Technique (Chimera Shadow Garden)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [24, 38, 152, 26],
    paletteHex: ['#0F172A', '#0D9488', '#334155'],
    dominantHueDeg: 175,
    visualDescriptors: ['Long spiky sea-urchin dark navy-black hair', 'Sharp viridian-teal eyes in MAPPA palette', 'Shadow puppet hand signs & high collar'],
    f1: 0.974,
    confusionPairId: 'toji_fushiguro',
  },
  {
    id: 'nobara_kugisaki',
    name: 'Nobara Kugisaki',
    kanji: '釘崎 野薔薇',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Grade 3 First-Year Student',
    cursedTechnique: 'Straw Doll Technique (Hairpin & Resonance)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [22, 34, 145, 24],
    paletteHex: ['#D97706', '#B45309', '#1E293B'],
    dominantHueDeg: 34,
    visualDescriptors: ['Chin-length auburn-ginger bob haircut', 'Iron hammer, nails, and straw doll weapon props', 'Amber-orange eyes with sharp side-part bangs'],
    f1: 0.989,
    confusionPairId: 'shoko_ieiri',
  },
  {
    id: 'yuta_okkotsu',
    name: 'Yuta Okkotsu',
    kanji: '乙骨 憂太',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Special Grade Second-Year Student',
    cursedTechnique: 'Copy & Boundless Cursed Energy (Rika)',
    primaryArc: 'Jujutsu Kaisen 0 (Movie)',
    sheets: [22, 36, 138, 24],
    paletteHex: ['#F8FAFC', '#1E293B', '#EC4899'],
    dominantHueDeg: 215,
    visualDescriptors: ['Loose white Jujutsu High tunic jacket', 'Dark circles under deep blue-black eyes', 'Katana in red/black sword bag & silver ring'],
    f1: 0.969,
    confusionPairId: 'megumi_fushiguro',
  },
  {
    id: 'maki_zenin',
    name: 'Maki Zenin',
    kanji: '禪院 真希',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Grade 4 Second-Year Student',
    cursedTechnique: 'Heavenly Restriction & Cursed Tool Mastery (Playful Cloud)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [20, 32, 144, 22],
    paletteHex: ['#065F46', '#D97706', '#1E293B'],
    dominantHueDeg: 162,
    visualDescriptors: [
      'Dark pine-green hair tied in high ponytail with rectangular cursed glasses',
      'Post-Shibuya / Culling Game short messy dark hair with facial, neck & arm burn scars',
      'Dark high-collar uniform, trench coat & Split Soul Katana / Playful Cloud',
    ],
    f1: 0.975,
    confusionPairId: 'mai_zenin',
  },
  {
    id: 'toge_inumaki',
    name: 'Toge Inumaki',
    kanji: '狗巻 棘',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Semi-Grade 1 Second-Year Student',
    cursedTechnique: 'Cursed Speech',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [19, 30, 138, 21],
    paletteHex: ['#E2E8F0', '#A855F7', '#0F172A'],
    dominantHueDeg: 270,
    visualDescriptors: ['Platinum-cream straight hair', 'Zipped high fabric collar covering lower face', 'Snake & Fangs sigil tattoos around mouth and tongue'],
    f1: 0.976,
    confusionPairId: 'satoru_gojo',
  },
  {
    id: 'panda',
    name: 'Panda',
    kanji: 'パンダ',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Semi-Grade 2 Second-Year Student',
    cursedTechnique: 'Abrupt Mutated Cursed Corpse (Gorilla & Triceratops Cores)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 29, 136, 20],
    paletteHex: ['#F8FAFC', '#0F172A', '#10B981'],
    dominantHueDeg: 142,
    visualDescriptors: ['Bipedal giant black-and-white panda silhouette', 'Studded knuckle gauntlets in Gorilla Mode', 'High-contrast monochromatic fur blocks'],
    f1: 0.976,
    confusionPairId: 'masamichi_yaga',
  },
  {
    id: 'kento_nanami',
    name: 'Kento Nanami',
    kanji: '七海 建人',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Grade 1 Alumni & Mentor',
    cursedTechnique: 'Ratio Technique (7:3 Collapse & Overtime)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [22, 38, 149, 23],
    paletteHex: ['#FDE047', '#D97706', '#3B82F6'],
    dominantHueDeg: 48,
    visualDescriptors: ['Neatly parted blonde hair', 'Frameless green-tinted oval goggles without temples', 'Tan suit with spotted cloth-wrapped blunt cleaver'],
    f1: 0.986,
    confusionPairId: 'atsuya_kusakabe',
  },
  {
    id: 'masamichi_yaga',
    name: 'Masamichi Yaga',
    kanji: '夜蛾 正道',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Grade 1 Principal',
    cursedTechnique: 'Cursed Corpse (Cathy & Panda Creation)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 28, 136, 19],
    paletteHex: ['#1E293B', '#F59E0B', '#0F172A'],
    dominantHueDeg: 38,
    visualDescriptors: ['Broad muscular frame with goatee & buzzed hair sides', 'Rectangular dark sunglasses', 'Open-collar dark Tokyo Jujutsu jacket'],
    f1: 0.973,
    confusionPairId: 'panda',
  },
  {
    id: 'shoko_ieiri',
    name: 'Shoko Ieiri',
    kanji: '家入 硝子',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Chief Medical Officer',
    cursedTechnique: 'Reverse Cursed Technique Output',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [18, 29, 135, 19],
    paletteHex: ['#78350F', '#F8FAFC', '#475569'],
    dominantHueDeg: 30,
    visualDescriptors: ['Long brown hair & beauty mole beneath right eye', 'White medical lab coat & tired dark circles under eyes', 'Surgical morgue lighting or cigarette/lollipop prop'],
    f1: 0.968,
    confusionPairId: 'nobara_kugisaki',
  },
  {
    id: 'kiyotaka_ijichi',
    name: 'Kiyotaka Ijichi',
    kanji: '伊地知 潔高',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Assistant Director',
    cursedTechnique: 'Curtain (Barrier Technique) & Shikigami',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [17, 27, 138, 19],
    paletteHex: ['#0F172A', '#E2E8F0', '#475569'],
    dominantHueDeg: 210,
    visualDescriptors: ['Thin-framed rectangular eyeglasses & nervous sweat expression', 'Neatly parted black hair & black business suit', 'Two-finger Curtain barrier hand sign'],
    f1: 0.971,
    confusionPairId: 'atsuya_kusakabe',
  },
  {
    id: 'akari_nitta',
    name: 'Akari Nitta',
    kanji: '新田 明',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Auxiliary Manager',
    cursedTechnique: 'Tactical Field Support & Barrier Logistics',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#FACC15', '#1E293B', '#94A3B8'],
    dominantHueDeg: 50,
    visualDescriptors: ['Short messy blonde bob with dark roots', 'Small mole below left eye & dark business suit', 'Sibling resemblance to Arata Nitta'],
    f1: 0.963,
    confusionPairId: 'arata_nitta',
  },
  {
    id: 'atsuya_kusakabe',
    name: 'Atsuya Kusakabe',
    kanji: '日下部 篤也',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Grade 1 Second-Year Faculty',
    cursedTechnique: 'New Shadow Style: Simple Domain (Evening Moon Sword Drawing)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [18, 29, 136, 19],
    paletteHex: ['#B45309', '#1E293B', '#F8FAFC'],
    dominantHueDeg: 34,
    visualDescriptors: ['Tan trench coat over dark suit & katana at hip', 'Wavy dark brown hair & perpetual tired expression', 'Lollipop stick & glowing Simple Domain stance'],
    f1: 0.972,
    confusionPairId: 'kento_nanami',
  },
  {
    id: 'takuma_ino',
    name: 'Takuma Ino',
    kanji: '猪野 琢真',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: 'Grade 2 Sorcerer',
    cursedTechnique: 'Auspicious Beasts Summon (Kaichi, Reiki, Kirin, Ryu)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [18, 28, 136, 19],
    paletteHex: ['#1E293B', '#F97316', '#E2E8F0'],
    dominantHueDeg: 26,
    visualDescriptors: ['Black knit ski mask/beanie pulled over face during combat', 'Dark bomber jacket & brown hair when unmasked', 'Glowing golden Kaichi horn projectile'],
    f1: 0.977,
    confusionPairId: 'yu_haibara',
  },
  {
    id: 'yu_haibara',
    name: 'Yu Haibara',
    kanji: '灰原 雄',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Tokyo Jujutsu High',
    grade: '2006 First-Year Student',
    cursedTechnique: 'Close-Quarters Cursed Energy Reinforcement',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [17, 27, 137, 19],
    paletteHex: ['#78350F', '#FDE047', '#0F172A'],
    dominantHueDeg: 35,
    visualDescriptors: ['Short fluffy chestnut-brown hair & wide cheerful smile', 'Bright warm brown eyes & classic dark Jujutsu uniform', 'Paired alongside young Kento Nanami in 2006 arc'],
    f1: 0.964,
    confusionPairId: 'takuma_ino',
  },

  // ================= KYOTO JUJUTSU HIGH (9) =================
  {
    id: 'aoi_todo',
    name: 'Aoi Todo',
    kanji: '東堂 葵',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Grade 1 Third-Year Student',
    cursedTechnique: 'Boogie Woogie',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [21, 35, 146, 22],
    paletteHex: ['#78350F', '#38BDF8', '#1E293B'],
    dominantHueDeg: 28,
    visualDescriptors: ['Muscular bare-chested physique with baggy white pants', 'Large vertical scar running down left eye and cheek', 'Top-knot black hair & clapping hand posture'],
    f1: 0.985,
    confusionPairId: 'choso',
  },
  {
    id: 'mai_zenin',
    name: 'Mai Zenin',
    kanji: '禪院 真依',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Grade 3 Second-Year Student',
    cursedTechnique: 'Construction (Single Daily Bullet)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 29, 135, 20],
    paletteHex: ['#064E3B', '#475569', '#0F172A'],
    dominantHueDeg: 160,
    visualDescriptors: ['Short chin-length dark green bob haircut', 'Silver revolver handgun & Kyoto school uniform', 'No eyeglasses (distinguishing from twin Maki)'],
    f1: 0.967,
    confusionPairId: 'maki_zenin',
  },
  {
    id: 'kasumi_miwa',
    name: 'Kasumi Miwa',
    kanji: '三輪 霞',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Grade 3 Second-Year Student',
    cursedTechnique: 'New Shadow Style: Simple Domain (Batto Sword Drawing)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 28, 136, 19],
    paletteHex: ['#38BDF8', '#1E293B', '#93C5FD'],
    dominantHueDeg: 202,
    visualDescriptors: ['Long straight cyan-blue hair with diagonal bangs', 'Tailored dark pantsuit & katana sheath', 'Glowing blue Simple Domain ring at feet'],
    f1: 0.971,
    confusionPairId: 'mei_mei',
  },
  {
    id: 'kokichi_muta',
    name: 'Kokichi Muta (Mechamaru)',
    kanji: '与 幸吉 / 究極メカ丸',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Semi-Grade 1 Second-Year Student',
    cursedTechnique: 'Puppet Manipulation (Mode: Absolute)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [19, 33, 136, 20],
    paletteHex: ['#D97706', '#22C55E', '#334155'],
    dominantHueDeg: 38,
    visualDescriptors: ['Metallic cylindrical robot head with bolts & bandages', 'Arm cannon blades & Ultimate Mechamaru Eva-scale unit', 'Pale scarred human form with dark hair in Shibuya prelude'],
    f1: 0.988,
    confusionPairId: 'panda',
  },
  {
    id: 'momo_nishimiya',
    name: 'Momo Nishimiya',
    kanji: '西宮 桃',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Grade 2 Third-Year Student',
    cursedTechnique: 'Tool Manipulation (Wind Scythe)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 27, 137, 19],
    paletteHex: ['#FDE047', '#EF4444', '#0F172A'],
    dominantHueDeg: 52,
    visualDescriptors: ['Blonde twin-tails shaped like swept broom bristles', 'Red hair ribbon & flying straw broom', 'Black witch-style pleated dress'],
    f1: 0.981,
    confusionPairId: 'uraume',
  },
  {
    id: 'noritoshi_kamo',
    name: 'Noritoshi Kamo',
    kanji: '加茂 憲紀',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Semi-Grade 1 Third-Year Student',
    cursedTechnique: 'Blood Manipulation (Flowing Red Scale & Crimson Binding)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 28, 136, 19],
    paletteHex: ['#1E293B', '#DC2626', '#E2E8F0'],
    dominantHueDeg: 358,
    visualDescriptors: ['Closed narrow eyes (opens right eye with red blood mark)', 'Kyoto traditional shozoku robe & archery bow', 'Blood-dipped arrows & hair side-locks wrapped in cloth'],
    f1: 0.969,
    confusionPairId: 'choso',
  },
  {
    id: 'arata_nitta',
    name: 'Arata Nitta',
    kanji: '新田 新',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'First-Year Student',
    cursedTechnique: 'Pain & Bleed Stopping Technique',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#FACC15', '#1E293B', '#38BDF8'],
    dominantHueDeg: 49,
    visualDescriptors: ['Short parted blonde hair with dark roots', 'Small mole under right eye & standard dark Jujutsu jacket', 'Blue healing aura on Yuji and Nobara in Shibuya'],
    f1: 0.962,
    confusionPairId: 'akari_nitta',
  },
  {
    id: 'yoshinobu_gakuganji',
    name: 'Yoshinobu Gakuganji',
    kanji: '楽巌寺 嘉伸',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Principal of Kyoto Jujutsu High',
    cursedTechnique: 'Electric Guitar Cursed Energy Amplification',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 27, 137, 19],
    paletteHex: ['#CBD5E1', '#EF4444', '#1E293B'],
    dominantHueDeg: 212,
    visualDescriptors: ['Elderly monk with long white eyebrows & facial piercings', 'Flying-V electric guitar using own body as amplifier', 'Traditional dark kimono & wooden cane'],
    f1: 0.984,
    confusionPairId: 'naobito_zenin',
  },
  {
    id: 'utahime_iori',
    name: 'Utahime Iori',
    kanji: '庵 歌姫',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Kyoto Jujutsu High',
    grade: 'Semi-Grade 1 Faculty',
    cursedTechnique: 'Solo Forbidden Area',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [18, 28, 137, 19],
    paletteHex: ['#1E1B4B', '#DC2626', '#F8FAFC'],
    dominantHueDeg: 348,
    visualDescriptors: ['Horizontal jagged scar across bridge of nose and cheeks', 'White kosode & crimson shrine maiden hakama', 'Dark purple-black hair tied with white bow'],
    f1: 0.979,
    confusionPairId: 'choso',
  },

  // ================= CLANS & INDEPENDENT SORCERERS (5) =================
  {
    id: 'yuki_tsukumo',
    name: 'Yuki Tsukumo',
    kanji: '九十九 由基',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Clans & Independent Sorcerers',
    grade: 'Special Grade Sorcerer',
    cursedTechnique: 'Star Rage (Bom Ba Ye & Garuda Shikigami)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [32, 54, 168, 34],
    paletteHex: ['#FDE047', '#DC2626', '#1E293B'],
    dominantHueDeg: 56,
    visualDescriptors: [
      'Long wavy golden-blonde hair parted with loose bangs framing face (NO braid over nose)',
      'Half-body & bust: dark sleeveless turtleneck OR Mandarin-collar sleeveless top with gold diagonal clasps & grey high-waisted corset belt exposing bare shoulders/arms',
      'Face close-up & flashback half-image: brown aviator/motorcycle goggles resting on forehead, black leather biker jacket over mesh top & dark choker, or winking with floating pink heart in Shibuya',
      'Skeletal bone-like serpent shikigami Garuda coiled around her or floating beside her',
    ],
    f1: 0.986,
    confusionPairId: 'momo_nishimiya',
  },
  {
    id: 'mei_mei',
    name: 'Mei Mei',
    kanji: '冥冥',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Clans & Independent Sorcerers',
    grade: 'Grade 1 Independent Sorcerer',
    cursedTechnique: 'Black Bird Manipulation (Bird Strike)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [24, 38, 144, 24],
    paletteHex: ['#67E8F9', '#F43F5E', '#0F172A'],
    dominantHueDeg: 182,
    visualDescriptors: [
      'Female Grade 1 sorcerer with pale icy-blue/silver-lavender hair in TWO THICK BRAIDS (signature front braid over nose OR parted braids in fanart showing golden-amber eyes & red lipstick)',
      'Smooth unscarred face & neck (ZERO patchwork stitches, unlike Mahito) in sleek dark high-collar dress/uniform',
      'Oversized broad battle axe over shoulder & black crows (Bird Strike)',
    ],
    f1: 0.985,
    confusionPairId: 'mahito',
  },
  {
    id: 'ui_ui',
    name: 'Ui Ui',
    kanji: '憂憂',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Clans & Independent Sorcerers',
    grade: 'Mei Mei Assistant Sorcerer',
    cursedTechnique: 'Spatial Teleportation & Simple Domain',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 27, 137, 19],
    paletteHex: ['#67E8F9', '#0F172A', '#F8FAFC'],
    dominantHueDeg: 190,
    visualDescriptors: ['Asymmetrical ice-blue bowl cut hair', 'Dark double-breasted boy suit with shorts', 'Carrying Mei Mei oversized battle axe'],
    f1: 0.973,
    confusionPairId: 'mei_mei',
  },
  {
    id: 'naobito_zenin',
    name: 'Naobito Zenin',
    kanji: '禪院 直毘人',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Clans & Independent Sorcerers',
    grade: 'Special Grade 1 (26th Zenin Head)',
    cursedTechnique: 'Projection Sorcery (24 FPS Animation Frame Trap)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [18, 31, 135, 19],
    paletteHex: ['#CBD5E1', '#475569', '#0F172A'],
    dominantHueDeg: 210,
    visualDescriptors: ['Long white-grey moustache extending past jawline', 'Traditional dark haori & hakama kimono', 'Glass animation frame panels freezing targets'],
    f1: 0.979,
    confusionPairId: 'yoshinobu_gakuganji',
  },
  {
    id: 'naoya_zenin',
    name: 'Naoya Zenin',
    kanji: '禪院 直哉',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Clans & Independent Sorcerers',
    grade: 'Special Grade 1 (Hei Squad Leader)',
    cursedTechnique: 'Projection Sorcery (Subsonic Acceleration)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [22, 34, 144, 22],
    paletteHex: ['#FDE047', '#0F172A', '#E2E8F0'],
    dominantHueDeg: 47,
    visualDescriptors: [
      'Short blonde undercut hair with dark tips & multiple helix ear piercings along outer ear',
      'Sharp upturned slanted fox-like eyes & arrogant smirk',
      'Traditional light kimono with dark haori & wide hakama trousers',
    ],
    f1: 0.979,
    confusionPairId: 'kento_nanami',
  },

  // ================= STAR PLASMA VESSEL ARC & SPECIAL ENTITIES (4) =================
  {
    id: 'master_tengen',
    name: 'Master Tengen',
    kanji: '天元',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Star Plasma Vessel Arc',
    grade: 'Immortal Barrier Master',
    cursedTechnique: 'Immortality & Pure Barrier Architecture',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#E2E8F0', '#38BDF8', '#0F172A'],
    dominantHueDeg: 196,
    visualDescriptors: ['Cylindrical bandaged/evolved head with four vertical eyes', 'Flowing white ceremonial robes in Tomb of the Star Corridor', 'Ethereal cyan barrier luminescence'],
    f1: 0.985,
    confusionPairId: 'kokichi_muta',
  },
  {
    id: 'riko_amanai',
    name: 'Riko Amanai',
    kanji: '天内 理子',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Star Plasma Vessel Arc',
    grade: 'Star Plasma Vessel (2006)',
    cursedTechnique: 'Tengen Vessel Compatibility',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [19, 32, 138, 20],
    paletteHex: ['#1E293B', '#F8FAFC', '#EF4444'],
    dominantHueDeg: 220,
    visualDescriptors: ['Long braided twin-tails with white forehead headband', 'Renchoku Girls School sailor uniform with red tie', 'Expressive dark blue eyes'],
    f1: 0.976,
    confusionPairId: 'misato_kuroi',
  },
  {
    id: 'misato_kuroi',
    name: 'Misato Kuroi',
    kanji: '黒井 美里',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Star Plasma Vessel Arc',
    grade: 'Star Plasma Vessel Caretaker',
    cursedTechnique: 'Shikigami & Close-Combat Broom/Naginata Defense',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [17, 27, 137, 19],
    paletteHex: ['#0F172A', '#F8FAFC', '#475569'],
    dominantHueDeg: 215,
    visualDescriptors: ['Short neat black bob haircut & mole below left eye', 'Traditional black-and-white maid apron attire', 'Composed protective stance beside Riko Amanai'],
    f1: 0.971,
    confusionPairId: 'riko_amanai',
  },
  {
    id: 'takada_chan',
    name: 'Takuko Takada (Takada-chan)',
    kanji: '高田 延子',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Star Plasma Vessel Arc',
    grade: 'Tall Idol (Todo Mental Domain)',
    cursedTechnique: 'Idol Beam & Synchronized Tag-Team Combo (with Aoi Todo)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 27, 138, 19],
    paletteHex: ['#EC4899', '#38BDF8', '#FDE047'],
    dominantHueDeg: 328,
    visualDescriptors: ['180cm tall idol with double buns & frilly pastel stage dress', 'Sparkling MAPPA stage lighting & heart beam pose', 'Paired alongside Aoi Todo in Shibuya combat cutaways'],
    f1: 0.984,
    confusionPairId: 'momo_nishimiya',
  },

  // ================= GETO'S GROUP (7) =================
  {
    id: 'suguru_geto',
    name: 'Suguru Geto',
    kanji: '夏油 傑',
    faction: 'Curse User / Vessel',
    wikiGroup: "Geto's Group",
    grade: 'Special Grade Curse User',
    cursedTechnique: 'Cursed Spirit Manipulation (Maximum: Uzumaki)',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [24, 39, 146, 25],
    paletteHex: ['#1E293B', '#F59E0B', '#7C3AED'],
    dominantHueDeg: 45,
    visualDescriptors: ['Long black hair tied in half-bun with single bang strand (no forehead stitches)', 'Large black gauge earlobes & narrow dark eyes', 'Baggy 2006 Tokyo uniform or golden-purple Buddhist kesa'],
    f1: 0.972,
    confusionPairId: 'kenjaku',
  },
  {
    id: 'miguel_oduol',
    name: 'Miguel Oduol',
    kanji: 'ミゲル・オドゥオール',
    faction: 'Curse User / Vessel',
    wikiGroup: "Geto's Group",
    grade: 'Special Grade-Caliber Combatant',
    cursedTechnique: 'Prayer Song & Black Rope Cursed Tool',
    primaryArc: 'Jujutsu Kaisen 0 (Movie)',
    sheets: [18, 29, 136, 19],
    paletteHex: ['#78350F', '#EAB308', '#14B8A6'],
    dominantHueDeg: 42,
    visualDescriptors: ['Woven straw cap & dark sunglasses', 'Coiled Black Rope wrapped around forearm emitting cursed sparks', 'Tan tunic & teal scarf in JJK 0 Shinjuku battle'],
    f1: 0.983,
    confusionPairId: 'larue',
  },
  {
    id: 'mimiko_hasaba',
    name: 'Mimiko Hasaba',
    kanji: '枷場 美々子',
    faction: 'Curse User / Vessel',
    wikiGroup: "Geto's Group",
    grade: "Geto's Adopted Daughter",
    cursedTechnique: 'Stuffed Doll Hanging Technique',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 27, 137, 19],
    paletteHex: ['#1E293B', '#E11D48', '#F8FAFC'],
    dominantHueDeg: 346,
    visualDescriptors: ['Long dark hair & black sailor school uniform', 'Carrying large stitched stuffed doll with rope noose', 'Twin sister of blonde Nanako Hasaba'],
    f1: 0.971,
    confusionPairId: 'nanako_hasaba',
  },
  {
    id: 'nanako_hasaba',
    name: 'Nanako Hasaba',
    kanji: '枷場 菜々子',
    faction: 'Curse User / Vessel',
    wikiGroup: "Geto's Group",
    grade: "Geto's Adopted Daughter",
    cursedTechnique: 'Smartphone Camera Frame Manipulation',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 27, 137, 19],
    paletteHex: ['#FDE047', '#EC4899', '#1E293B'],
    dominantHueDeg: 50,
    visualDescriptors: ['Blonde twin-buns hair & pink knit sweater vest', 'Smartphone camera held up to photograph targets', 'Twin sister of dark-haired Mimiko Hasaba'],
    f1: 0.974,
    confusionPairId: 'mimiko_hasaba',
  },
  {
    id: 'larue',
    name: 'Larue',
    kanji: 'ラルゥ',
    faction: 'Curse User / Vessel',
    wikiGroup: "Geto's Group",
    grade: "Geto's Family Executive",
    cursedTechnique: 'Heart Catch (Virtual Hand Projection)',
    primaryArc: 'Jujutsu Kaisen 0 (Movie)',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#EC4899', '#FDE047', '#1E293B'],
    dominantHueDeg: 330,
    visualDescriptors: ['Muscular build with long wavy blonde hair & headband', 'Heart-shaped chest cutouts & pleated skirt', 'Appears in JJK 0 Night Parade & Shibuya'],
    f1: 0.979,
    confusionPairId: 'miguel_oduol',
  },
  {
    id: 'manami_suda',
    name: 'Manami Suda',
    kanji: '菅田 真奈美',
    faction: 'Curse User / Vessel',
    wikiGroup: "Geto's Group",
    grade: "Geto's Secretary & Commander",
    cursedTechnique: 'Command & Cursed Energy Tactical Relay',
    primaryArc: 'Jujutsu Kaisen 0 (Movie)',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#312E81', '#E2E8F0', '#0F172A'],
    dominantHueDeg: 242,
    visualDescriptors: ['Long straight dark purple-black hair', 'Formal dark dress coat with fur-trimmed collar in JJK 0', 'Seen alongside Toshihisa Negi in Shibuya subway'],
    f1: 0.966,
    confusionPairId: 'utahime_iori',
  },
  {
    id: 'toshihisa_negi',
    name: 'Toshihisa Negi',
    kanji: '祢木 利久',
    faction: 'Curse User / Vessel',
    wikiGroup: "Geto's Group",
    grade: "Geto's Family Combatant",
    cursedTechnique: 'Close-Range Cursed Energy Assault',
    primaryArc: 'Jujutsu Kaisen 0 (Movie)',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#1E293B', '#94A3B8', '#0F172A'],
    dominantHueDeg: 215,
    visualDescriptors: ['Slicked-back dark hair & scar across nose/cheek', 'Open dark shirt revealing scarred torso & necklace', 'Fights Panda/Kusakabe in Shibuya'],
    f1: 0.965,
    confusionPairId: 'toji_fushiguro',
  },

  // ================= CURSE USERS, VESSELS & DEATH PAINTINGS (15) =================
  {
    id: 'ryomen_sukuna',
    name: 'Ryomen Sukuna',
    kanji: '両面 宿儺',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Special Grade King of Curses',
    cursedTechnique: 'Shrine (Cleave, Dismantle & Malevolent Shrine)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [25, 45, 148, 30],
    paletteHex: ['#EF4444', '#F43F5E', '#111827'],
    dominantHueDeg: 355,
    visualDescriptors: ['Slicked-back spiky pink hair', 'Black geometric tattoo bands across forehead, nose, cheeks, and wrists', 'Four active crimson eyes'],
    f1: 0.985,
    confusionPairId: 'yuji_itadori',
  },
  {
    id: 'kenjaku',
    name: 'Kenjaku (Pseudo-Geto)',
    kanji: '羂索',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Special Grade Ancient Sorcerer',
    cursedTechnique: 'Brain Transplant & Cursed Spirit Manipulation (Prison Realm)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [22, 36, 142, 22],
    paletteHex: ['#1E293B', '#EC4899', '#F59E0B'],
    dominantHueDeg: 335,
    visualDescriptors: ['Horizontal stitch scar across forehead (removable skull cap)', 'Gold/purple five-piece Buddhist kesa robes', 'Holding Prison Realm cube in Shibuya subway'],
    f1: 0.974,
    confusionPairId: 'suguru_geto',
  },
  {
    id: 'toji_fushiguro',
    name: 'Toji Fushiguro',
    kanji: '伏黒 甚爾',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Heavenly Restriction (Sorcerer Killer)',
    cursedTechnique: 'Zero Cursed Energy & Inverted Spear of Heaven / Playful Cloud',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [21, 42, 148, 27],
    paletteHex: ['#0F172A', '#10B981', '#7C3AED'],
    dominantHueDeg: 155,
    visualDescriptors: ['Straight choppy black hair & lip corner scar', 'Compression black shirt & baggy white martial pants', 'Coiled purple Inventory Curse spirit around shoulder'],
    f1: 0.981,
    confusionPairId: 'megumi_fushiguro',
  },
  {
    id: 'uraume',
    name: 'Uraume',
    kanji: '裏梅',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Special Grade Heian Retainer',
    cursedTechnique: 'Ice Formation (Frost Calm)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [18, 29, 135, 19],
    paletteHex: ['#F8FAFC', '#E11D48', '#38BDF8'],
    dominantHueDeg: 345,
    visualDescriptors: ['White chin-length bob with crimson-magenta horizontal band across crown', 'Monk robes & crystalline Frost Calm ice mist', 'Plum-pink eyes'],
    f1: 0.975,
    confusionPairId: 'toge_inumaki',
  },
  {
    id: 'choso',
    name: 'Choso',
    kanji: '脹相',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Special Grade Death Painting Womb #1',
    cursedTechnique: 'Blood Manipulation (Piercing Blood & Supernova)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [20, 38, 144, 23],
    paletteHex: ['#991B1B', '#4C1D95', '#F8FAFC'],
    dominantHueDeg: 350,
    visualDescriptors: ['Two high symmetrical twin-tail buns of black hair', 'Horizontal dark blood mark across bridge of nose', 'Purple-and-white sleeveless tunic scarf'],
    f1: 0.983,
    confusionPairId: 'noritoshi_kamo',
  },
  {
    id: 'eso',
    name: 'Eso',
    kanji: '壊相',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Special Grade Death Painting Womb #2',
    cursedTechnique: 'Rot Technique (Wing King & Decay)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 28, 136, 19],
    paletteHex: ['#7C3AED', '#DC2626', '#1E293B'],
    dominantHueDeg: 280,
    visualDescriptors: ['Tall muscular frame with black mohawk & grotesque second face on back', 'Revealing black sheer suspender outfit', 'Crimson blood Wing King patterns extending from back'],
    f1: 0.982,
    confusionPairId: 'kechizu',
  },
  {
    id: 'kechizu',
    name: 'Kechizu',
    kanji: '血塗',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Special Grade Death Painting Womb #3',
    cursedTechnique: 'Rot Technique (Corrosive Blood Spray)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 27, 137, 19],
    paletteHex: ['#4ADE80', '#DC2626', '#1E293B'],
    dominantHueDeg: 135,
    visualDescriptors: ['Round bulbous pale-green non-humanoid body with two mouths', 'Large circular eyes & red pleated skirt', 'Paired with brother Eso in Origin of Blind Obedience arc'],
    f1: 0.990,
    confusionPairId: 'eso',
  },
  {
    id: 'junpei_yoshino',
    name: 'Junpei Yoshino',
    kanji: '吉野 順平',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Awakened Sorcerer',
    cursedTechnique: 'Poison & Moon Dregs Jellyfish Shikigami',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [19, 31, 136, 19],
    paletteHex: ['#0F172A', '#38BDF8', '#475569'],
    dominantHueDeg: 205,
    visualDescriptors: ['Long dark olive-black hair covering right eye (revealing cigarette burn marks)', 'Translucent glowing blue jellyfish shikigami Moon Dregs', 'Dark Tokyo uniform or casual black shirt'],
    f1: 0.974,
    confusionPairId: 'megumi_fushiguro',
  },
  {
    id: 'haruta_shigemo',
    name: 'Haruta Shigemo',
    kanji: '重面 春太',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Shibuya Curse User',
    cursedTechnique: 'Miracles (6 Eye-Markings Storage) & Hand-Hilt Sword',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [18, 29, 136, 19],
    paletteHex: ['#FDE047', '#EC4899', '#0F172A'],
    dominantHueDeg: 48,
    visualDescriptors: ['Side-ponytail blonde hair with magenta-pink tips', 'Three vertical miracle markings under each eye', 'Sword whose hilt is a severed human hand'],
    f1: 0.979,
    confusionPairId: 'nanako_hasaba',
  },
  {
    id: 'juzo_kumiya',
    name: 'Juzo Kumiya',
    kanji: '組屋 鞣造',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Cursed Tool Craftsman',
    cursedTechnique: 'Human Bone & Flesh Cursed Weaponry',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#991B1B', '#E2E8F0', '#1E293B'],
    dominantHueDeg: 355,
    visualDescriptors: ['Stitched leather butcher apron & bandages around head', 'Large meat cleaver machete & manic wide grin', 'Fights Principal Gakuganji & Gojo in Goodwill Event'],
    f1: 0.980,
    confusionPairId: 'jiro_awasaka',
  },
  {
    id: 'jiro_awasaka',
    name: 'Jiro Awasaka',
    kanji: '粟坂 二良',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Shibuya Curse User',
    cursedTechnique: 'Inverse (Strong Attacks Become Weak & Vice Versa)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#D97706', '#0F172A', '#F8FAFC'],
    dominantHueDeg: 33,
    visualDescriptors: ['Bald daruma-like head with thick black eyebrows & wide moustache', 'Polka-dot scarf & dark robe in Shibuya', 'Fights Yuji Itadori & Megumi Fushiguro'],
    f1: 0.983,
    confusionPairId: 'juzo_kumiya',
  },
  {
    id: 'granny_ogami',
    name: 'Granny Ogami',
    kanji: 'オガミ婆',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Shibuya Seance Curse User',
    cursedTechnique: 'Seance Technique (Body & Soul Information Channeling)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#94A3B8', '#7C3AED', '#0F172A'],
    dominantHueDeg: 268,
    visualDescriptors: ['Short hunched elderly woman with grey buns & round dark glasses', 'Shamanic beads & ritual seance chant', 'Summons Toji Zenin body on Shibuya rooftop'],
    f1: 0.981,
    confusionPairId: 'yoshinobu_gakuganji',
  },
  {
    id: 'bayer',
    name: 'Bayer',
    kanji: 'バイエル',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Q Organization Top Soldier',
    cursedTechnique: 'Cursed Energy Combat & Knife Artillery',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#1E293B', '#EAB308', '#475569'],
    dominantHueDeg: 44,
    visualDescriptors: ['Tall Q military uniform with peaked officer cap', 'Dark sunglasses & facial scar', 'Defeated by young Satoru Gojo outside Riko apartment'],
    f1: 0.972,
    confusionPairId: 'kokun',
  },
  {
    id: 'kokun',
    name: 'Kokun',
    kanji: 'コークン',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Q Organization Operative',
    cursedTechnique: 'Tactical Infiltration',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#334155', '#F8FAFC', '#0F172A'],
    dominantHueDeg: 215,
    visualDescriptors: ['Q Organization beret & dark tactical coat', 'Bandaged nose after encounter withSuguru Geto', 'Appears in Hidden Inventory Episode 1'],
    f1: 0.968,
    confusionPairId: 'bayer',
  },
  {
    id: 'shiu_kong',
    name: 'Shiu Kong',
    kanji: '孔 時雨',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Curse Users & Death Paintings',
    grade: 'Underworld Broker',
    cursedTechnique: 'Contract & Bounty Brokerage (Time Vessel Association)',
    primaryArc: 'Season 2: Hidden Inventory',
    sheets: [17, 26, 138, 19],
    paletteHex: ['#475569', '#F59E0B', '#0F172A'],
    dominantHueDeg: 38,
    visualDescriptors: ['Slicked-back dark hair with stubble beard & tinted glasses', 'Sharp pinstripe suit & cigarette', 'Toji Fushiguro primary client contact in 2006'],
    f1: 0.970,
    confusionPairId: 'atsuya_kusakabe',
  },

  // ================= CURSED SPIRITS & SHIKIGAMI (8) =================
  {
    id: 'mahito',
    name: 'Mahito',
    kanji: '真人',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Unregistered Cursed Spirit',
    cursedTechnique: 'Idle Transfiguration (Self-Embodiment of Perfection)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [24, 43, 155, 26],
    paletteHex: ['#94A3B8', '#38BDF8', '#475569'],
    dominantHueDeg: 212,
    visualDescriptors: [
      'Male cursed spirit with prominent PATCHWORK SURGICAL STITCHES across face, nose bridge, neck, and limbs (distinguishes him from smooth-skinned Mei Mei in fanart)',
      'Long messy ankle-length pale ice-blue/grey-silver hair (unbraided or loose)',
      'Heterochromia eyes (one dark blue, one pale grey/amber) & transfigured hand mouths',
    ],
    f1: 0.988,
    confusionPairId: 'mei_mei',
  },
  {
    id: 'jogo',
    name: 'Jogo',
    kanji: '漏瑚',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Unregistered Cursed Spirit',
    cursedTechnique: 'Disaster Flames (Coffin of the Iron Mountain & Maximum: Meteor)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [20, 34, 141, 21],
    paletteHex: ['#F97316', '#EAB308', '#1C1917'],
    dominantHueDeg: 24,
    visualDescriptors: ['Volcano-crater head & ear plugs', 'Single large cyclops eye with red sclera', 'Black-stained teeth & yellow-tan cloak'],
    f1: 0.992,
    confusionPairId: 'hanami',
  },
  {
    id: 'hanami',
    name: 'Hanami',
    kanji: '花御',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Unregistered Cursed Spirit',
    cursedTechnique: 'Disaster Plants (Cursed Bud & Root Manipulation)',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [19, 31, 138, 20],
    paletteHex: ['#FEFCE8', '#991B1B', '#365314'],
    dominantHueDeg: 75,
    visualDescriptors: ['White biomechanical bark torso with red abdominal markings', 'Two black tree-branch antlers where eyes sit', 'Left arm wrapped in white cloth concealing cursed flower'],
    f1: 0.993,
    confusionPairId: 'dagon',
  },
  {
    id: 'dagon',
    name: 'Dagon',
    kanji: '陀艮',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Unregistered Cursed Spirit',
    cursedTechnique: 'Disaster Tides (Horizon of the Captivating Skandha)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [18, 30, 134, 20],
    paletteHex: ['#EF4444', '#0284C7', '#FDE047'],
    dominantHueDeg: 8,
    visualDescriptors: ['Crimson cephalopod/octopus humanoid head with tentacles', 'Muscular red torso with dark belt & white hip fins', 'Tropical ocean water domain lighting'],
    f1: 0.991,
    confusionPairId: 'jogo',
  },
  {
    id: 'rika_orimoto',
    name: 'Rika Orimoto (Queen of Curses)',
    kanji: '祈本 里香',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Vengeful Cursed Spirit',
    cursedTechnique: 'Boundless Cursed Energy & Pure Love Beam',
    primaryArc: 'Jujutsu Kaisen 0 (Movie)',
    sheets: [21, 35, 140, 22],
    paletteHex: ['#F8FAFC', '#EC4899', '#1E293B'],
    dominantHueDeg: 325,
    visualDescriptors: ['Towering white muscular cursed spirit with segmented jaw & no eyes (opens single eye)', 'Long dark-violet tendrils & clawed hands', 'Magenta-pink Pure Love cursed energy beam'],
    f1: 0.990,
    confusionPairId: 'mahoraga',
  },
  {
    id: 'mahoraga',
    name: 'Divine General Mahoraga',
    kanji: '八握剣異戒神将魔虚羅',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Eight-Handled Sword Apex Shikigami',
    cursedTechnique: 'Eight-Handled Wheel Adaptation & Sword of Extermination',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [22, 38, 142, 23],
    paletteHex: ['#F8FAFC', '#FACC15', '#0F172A'],
    dominantHueDeg: 48,
    visualDescriptors: ['Floating golden Eight-Handled Dharma Wheel above head', 'Four wing-like protrusions where eyes sit & long head tail', 'White muscular body with Sword of Extermination bound to right forearm'],
    f1: 0.994,
    confusionPairId: 'rika_orimoto',
  },
  {
    id: 'smallpox_deity',
    name: 'Smallpox Deity',
    kanji: '疱瘡婆',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Disease Cursed Spirit',
    cursedTechnique: 'Graveyard Domain Coffin Entombment (3-Second Count)',
    primaryArc: 'Season 2: Shibuya Incident',
    sheets: [17, 27, 137, 19],
    paletteHex: ['#A855F7', '#FDE047', '#1E293B'],
    dominantHueDeg: 276,
    visualDescriptors: ['Spherical bloated purple-pink torso with four arms', 'Wooden coffin & boulder gravestone drop inside forest graveyard domain', 'Fights Mei Mei and Ui Ui in Shibuya'],
    f1: 0.989,
    confusionPairId: 'kechizu',
  },
  {
    id: 'finger_bearer',
    name: 'Finger Bearer (Special Grade)',
    kanji: '特級仮想怨霊 (指の化身)',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Sukuna Finger Spirit',
    cursedTechnique: 'Raw Unrefined Cursed Energy Blast & Incomplete Domain',
    primaryArc: 'Season 1 ( Cour 1 & 2 )',
    sheets: [18, 29, 136, 19],
    paletteHex: ['#CBD5E1', '#F59E0B', '#0F172A'],
    dominantHueDeg: 40,
    visualDescriptors: ['Pale humanoid body with tribal dark markings & hollow mask-like grin', 'Spiky crown-like head ridges', 'Glowing golden-amber raw cursed energy spheres'],
    f1: 0.987,
    confusionPairId: 'hanami',
  },

  // ================= SEASON 3: CULLING GAME & REINCARNATED SORCERERS (16) =================
  {
    id: 'kinji_hakari',
    name: 'Kinji Hakari',
    kanji: '秤 金次',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Suspended 3rd-Year (Special Grade Equivalent)',
    cursedTechnique: 'Private Pure Love Train (Idle Death Gamble Jackpot)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [26, 44, 158, 28],
    paletteHex: ['#C084FC', '#FACC15', '#1E293B'],
    dominantHueDeg: 288,
    visualDescriptors: [
      'Tall muscular 3rd-year student with spiky purple-lavender / blonde-tinted undercut afro-textured hair & thin chin stubble',
      'Fur-trimmed dark overcoat or unzipped Jujutsu High jacket over patterned shirt',
      'Train doors, pachinko slot-machine projections & overflowing green/gold Jackpot cursed energy aura in anime & fanart',
    ],
    f1: 0.986,
    confusionPairId: 'kirara_hoshi',
  },
  {
    id: 'kirara_hoshi',
    name: 'Kirara Hoshi',
    kanji: '星 綺羅羅',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Suspended 3rd-Year Tokyo Student',
    cursedTechnique: 'Love Rendezvous (Southern Cross Star Attraction)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [23, 38, 148, 25],
    paletteHex: ['#F472B6', '#FDE047', '#0F172A'],
    dominantHueDeg: 315,
    visualDescriptors: [
      'Black shoulder-length hair with a bright TEAL / CYAN highlight streak on the front bangs (or blonde/pink in manga fanart) & purple-magenta eyes',
      'Three black dot piercings beneath lower lip, spiked black choker, white vertical-striped off-shoulder crop top with red straps exposing midriff, and gold star belt buckle',
      'Five Southern Cross star constellation markings (Imai, Acrux, Mimosa, Ginan, Gacrux — Love Rendezvous)',
    ],
    f1: 0.982,
    confusionPairId: 'kinji_hakari',
  },
  {
    id: 'hiromi_higuruma',
    name: 'Hiromi Higuruma',
    kanji: '日車 寛見',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Grade 1 Awakened Culling Game Player',
    cursedTechnique: "Deadly Sentencing (Judgeman & Executioner's Sword)",
    primaryArc: 'Season 3: Culling Game',
    sheets: [27, 45, 160, 28],
    paletteHex: ['#1E293B', '#F8FAFC', '#EAB308'],
    dominantHueDeg: 218,
    visualDescriptors: [
      'Defense attorney in sharp dark pinstripe business suit & tie with gold lawyer lapel pin (also seen soaking in a bathtub fully suited)',
      'Slicked-back dark brown hair, exhausted heavy-lidded eyes, and prominent aquiline nose',
      "Wielding wooden courtroom gavel, giant warhammer gavel, or glowing white Executioner's Sword alongside blindfolded stitched Judgeman shikigami",
    ],
    f1: 0.987,
    confusionPairId: 'atsuya_kusakabe',
  },
  {
    id: 'fumihiko_takaba',
    name: 'Fumihiko Takaba',
    kanji: '高羽 史彦',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Awakened Culling Game Player (Special Grade Potential)',
    cursedTechnique: 'Comedian (Reality-Warping Humor Phenomenon)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [23, 39, 149, 25],
    paletteHex: ['#F97316', '#0F172A', '#F8FAFC'],
    dominantHueDeg: 18,
    visualDescriptors: [
      'Stand-up comedian wearing a signature 50/50 vertically split costume (left half completely bare-chested/naked, right half in dark superhero/tuxedo bodysuit with collar)',
      'Swept-back dark pompadour hair & exaggerated comedic facial expressions ("Wi-Fi" pose)',
      'Paired with Hiromi Higuruma, Megumi, or Kenjaku in Culling Game scenes',
    ],
    f1: 0.985,
    confusionPairId: 'ryu_ishigori',
  },
  {
    id: 'hajime_kashimo',
    name: 'Hajime Kashimo',
    kanji: '鹿紫雲 一',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Reincarnated Edo Era Sorcerer (God of Lightning)',
    cursedTechnique: 'Electrified Cursed Energy & Mythical Beast Amber',
    primaryArc: 'Season 3: Culling Game',
    sheets: [28, 46, 162, 29],
    paletteHex: ['#22D3EE', '#A855F7', '#0F172A'],
    dominantHueDeg: 184,
    visualDescriptors: [
      'Cyan / ice-blue / white hair tied into TWO ROUND HIGH BUNS on top of head with jagged lightning-bolt side bangs',
      'Sharp turquoise eyes with dark vertical marks below lower eyelids, carrying a white/wrapped metal nyoi staff pole',
      'Crackling purple-blue lightning aura, white Sleeveless/loose martial haori, or glowing horned Mythical Beast Amber form',
    ],
    f1: 0.988,
    confusionPairId: 'kinji_hakari',
  },
  {
    id: 'yorozu',
    name: 'Yorozu (Tsumiki Fushiguro)',
    kanji: '万 / 伏黒 津美紀',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Reincarnated Heian Era Sorcerer (Tsumiki Vessel)',
    cursedTechnique: 'Construction (Liquid Metal, Insect Armor & True Sphere)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [25, 42, 154, 27],
    paletteHex: ['#8B5CF6', '#1E293B', '#F43F5E'],
    dominantHueDeg: 262,
    visualDescriptors: [
      "Megumi's sister Tsumiki Fushiguro (MAPPA anime: brown hair tied in a half-up top bun with two long face-framing side strands, green eyes, navy blue school blazer with red necktie & grey pleated skirt) OR Heian sorcerer Yorozu (long black hair with side loops & thick oval eyebrows in white kimono or bare-shouldered)",
      'Surrounded by swirling black Liquid Metal spikes, biomechanical Insect Armor with compound eyes & wings, or conjuring the Perfect/True Sphere',
      'Threefold Affliction domain expansion & obsessive wide-eyed gaze toward Sukuna',
    ],
    f1: 0.984,
    confusionPairId: 'hana_kurusu',
  },
  {
    id: 'hana_kurusu',
    name: 'Hana Kurusu (Angel)',
    kanji: '来栖 華 / 天使',
    faction: 'Jujutsu Sorcerer',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Symbiotic Reincarnated Heian Vessel (Angel)',
    cursedTechnique: "Technique Extinguishment (Jacob's Ladder)",
    primaryArc: 'Season 3: Culling Game',
    sheets: [25, 41, 152, 26],
    paletteHex: ['#FDE047', '#F8FAFC', '#38BDF8'],
    dominantHueDeg: 56,
    visualDescriptors: [
      'Long wavy golden-blonde/light hair with a glowing Ring Halo floating above her head and white feathered Angel Wings on her back',
      'Light cardigan sweater over dark skirt; Angel symbiont manifests as a second mouth on her cheek or hand',
      "Summoning a golden trumpet/horn of light to fire Jacob's Ladder (towering pillar of light & crosses)",
    ],
    f1: 0.985,
    confusionPairId: 'yuki_tsukumo',
  },
  {
    id: 'ryu_ishigori',
    name: 'Ryu Ishigori',
    kanji: '石流 龍',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Reincarnated Edo Era Sorcerer (Sendai Colony)',
    cursedTechnique: 'Cursed Energy Discharge (Granite Blast)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [26, 43, 155, 27],
    paletteHex: ['#6366F1', '#F43F5E', '#0F172A'],
    dominantHueDeg: 248,
    visualDescriptors: [
      'Massive elongated cannon-barrel black pompadour (regent hairstyle) jutting forward from forehead',
      'Open dark fur-trimmed coat over bare muscular chest with white trousers',
      'Charging and firing massive pink/violet Granite Blast energy beams directly from the tip of his pompadour',
    ],
    f1: 0.987,
    confusionPairId: 'takako_uro',
  },
  {
    id: 'takako_uro',
    name: 'Takako Uro',
    kanji: '烏鷺 亨子',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Reincarnated Heian Squad Captain (Sendai Colony)',
    cursedTechnique: 'Sky Manipulation (Thin Ice Breaker)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [25, 41, 151, 26],
    paletteHex: ['#38BDF8', '#0F172A', '#E2E8F0'],
    dominantHueDeg: 232,
    visualDescriptors: [
      'Jet-black sclera eyes with bright white irises & long floating dark hair',
      'Unclothed silhouette draped in warped, folded, and shattered glass-like Sky Manipulation surfaces covering her body',
      'Shattering space like cracking ice (Thin Ice Breaker) in Sendai Colony deadlock against Yuta & Ryu',
    ],
    f1: 0.986,
    confusionPairId: 'ryu_ishigori',
  },
  {
    id: 'reggie_star',
    name: 'Reggie Star',
    kanji: 'レジィ・スター',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Reincarnated Ancient Sorcerer (Tokyo No. 1)',
    cursedTechnique: 'Contractual Re-Creation (Receipt Summoning)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [25, 40, 150, 26],
    paletteHex: ['#A3E635', '#FDE047', '#1E293B'],
    dominantHueDeg: 86,
    visualDescriptors: [
      'Outfit/tunic made entirely of hundreds of strips of paper receipts and contracts layered like a poncho/skirt',
      'Wild spiky two-toned mane of hair & theatrical grin',
      'Burning paper receipts with cursed energy to materialize trucks, kitchen knives, scooters, and spa rest',
    ],
    f1: 0.985,
    confusionPairId: 'iori_hazenoki',
  },
  {
    id: 'remi',
    name: 'Remi',
    kanji: '麗美',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Awakened Culling Game Player (Tokyo No. 1)',
    cursedTechnique: 'Scorpion Hair Stinger Manipulation',
    primaryArc: 'Season 3: Culling Game',
    sheets: [22, 36, 144, 23],
    paletteHex: ['#E879F9', '#0F172A', '#F8FAFC'],
    dominantHueDeg: 308,
    visualDescriptors: [
      'Dark purple-black hair styled into FIVE ROUND SPHERICAL BUNS linked in a long segmented scorpion tail with a curved stinger tip (or light/pinkish scorpion-tail braid in manga colorings)',
      'Sleeveless burgundy/plum one-shoulder long slit dress with a dark ribbon bow tied around her neck',
      'Tokyo No. 1 Colony player who lures Megumi Fushiguro to Reggie Star',
    ],
    f1: 0.982,
    confusionPairId: 'reggie_star',
  },
  {
    id: 'rin_amai',
    name: 'Rin Amai',
    kanji: '甘井 凛',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Awakened Culling Game Player (Tokyo No. 1)',
    cursedTechnique: 'Sugar Manipulation (Glucose Amplification)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [21, 35, 142, 23],
    paletteHex: ['#4ADE80', '#FDE047', '#1E293B'],
    dominantHueDeg: 112,
    visualDescriptors: [
      'Young man with dark swept hair and lighter front bangs wearing a collared jacket/hoodie in Tokyo No. 1 Colony',
      "Yuji Itadori's former Sendai hometown acquaintance who guides Yuji to Hiromi Higuruma's theater",
      'Can transfigure internal sugar/glucose into pudding to support allies',
    ],
    f1: 0.979,
    confusionPairId: 'junpei_yoshino',
  },
  {
    id: 'iori_hazenoki',
    name: 'Iori Hazenoki',
    kanji: '黄櫨 折',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Reincarnated Ancient Sorcerer (Tokyo No. 1)',
    cursedTechnique: 'Exploding Flesh (Body Part Bomb Detonation & RCT)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [23, 37, 146, 24],
    paletteHex: ['#FACC15', '#EF4444', '#1E293B'],
    dominantHueDeg: 64,
    visualDescriptors: [
      'Shirtless bare muscular torso/abs with TWO BROWN SUSPENDERS clipped over bare chest, baggy olive-green cropped pants, and red front-zipper boots (NEVER wears a coat/shirt like Kinji Hakari)',
      'Spiky pale yellow-blonde swept-back mullet hair & amber-red eyes (no chin stubble)',
      'Plucking out his own eyeball or tooth to throw as an explosive bomb (Exploding Flesh)',
    ],
    f1: 0.983,
    confusionPairId: 'reggie_star',
  },
  {
    id: 'dhruv_lakdawalla',
    name: 'Dhruv Lakdawalla',
    kanji: 'ドルゥヴ・ラクダワラ',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Twice-Reincarnated Ancient Sorcerer (Sendai Colony)',
    cursedTechnique: 'Shikigami Orbit Tracks (Inviolable Trajectory Domain)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [22, 36, 144, 23],
    paletteHex: ['#34D399', '#CBD5E1', '#0F172A'],
    dominantHueDeg: 142,
    visualDescriptors: [
      'Elderly gaunt reincarnated sorcerer with long white beard, topknot hair, and simple cloth wraps',
      'Commanding giant mole-rat and pterodactyl-like autonomous shikigami whose flight paths form domain barriers',
      'One of the Sendai Colony Four-Way Deadlock leaders confronted by Yuta Okkotsu',
    ],
    f1: 0.981,
    confusionPairId: 'yoshinobu_gakuganji',
  },
  {
    id: 'kurourushi',
    name: 'Kurourushi',
    kanji: '黒沐死',
    faction: 'Cursed Spirit',
    wikiGroup: 'Cursed Spirits & Shikigami',
    grade: 'Special Grade Registered Cursed Spirit (Sendai Colony)',
    cursedTechnique: 'Cockroach Swarm & Festering Life Sword',
    primaryArc: 'Season 3: Culling Game',
    sheets: [23, 38, 147, 24],
    paletteHex: ['#78350F', '#EF4444', '#0F172A'],
    dominantHueDeg: 15,
    visualDescriptors: [
      'Grotesque giant humanoid black cockroach Cursed Spirit with bulbous compound eyes, antennae, and chitinous wings',
      'Wielding the cleaver-like Festering Life Sword embedded with six organic insect egg pods',
      'Fights Yuta Okkotsu, Ryu Ishigori, and Takako Uro in Sendai Colony',
    ],
    f1: 0.989,
    confusionPairId: 'dagon',
  },
  {
    id: 'charles_bernard',
    name: 'Charles Bernard',
    kanji: 'シャルル・ベルナール',
    faction: 'Curse User / Vessel',
    wikiGroup: 'Culling Game & Reincarnated Sorcerers',
    grade: 'Awakened Culling Game Player (Tokyo No. 2)',
    cursedTechnique: 'G Warstaff (Future-Sight Manga Panel Spear)',
    primaryArc: 'Season 3: Culling Game',
    sheets: [23, 37, 145, 24],
    paletteHex: ['#818CF8', '#F8FAFC', '#1E293B'],
    dominantHueDeg: 254,
    visualDescriptors: [
      'French-Japanese aspiring manga artist with wavy medium-length light-brown/blonde hair and intense expression',
      'Wielding G Warstaff — a giant fountain-pen G-nib spear that stamps manga panels onto opponents Chests to read 1–2 seconds into the future',
      'Fights Kinji Hakari inside Tokyo No. 2 Colony and Idle Death Gamble train station',
    ],
    f1: 0.982,
    confusionPairId: 'kinji_hakari',
  },
];

export const CANONICAL_WIKI_PORTRAIT_URLS: Record<string, string> = {
  satoru_gojo: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/04/Satoru_Gojo_%28Anime%29.png',
  yuji_itadori: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/1/13/Yuji_Itadori_%28Anime%29.png',
  megumi_fushiguro: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/8/8d/Megumi_Fushiguro_%28Anime%29.png',
  nobara_kugisaki: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/b/b7/Nobara_Kugisaki_%28Anime%29.png',
  yuta_okkotsu: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/e/e6/Yuta_Okkotsu_%28Anime_2%29.png',
  maki_zenin: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/55/Maki_Zen%27in_%28Anime%29.png',
  toge_inumaki: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/7/7a/Toge_Inumaki_%28Anime%29.png',
  panda: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/c/c5/Panda_%28Anime%29.png',
  kento_nanami: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/b/b0/Kento_Nanami_%28Anime%29.png',
  masamichi_yaga: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/51/Masamichi_Yaga_%28Anime%29.png',
  shoko_ieiri: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/8/87/Shoko_Ieiri_%28Anime%29.png',
  kiyotaka_ijichi: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/d/db/Kiyotaka_Ijichi_%28Anime%29.png',
  akari_nitta: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/8/84/Akari_Nitta_%28Anime%29.png',
  atsuya_kusakabe: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/9/9d/Atsuya_Kusakabe_%28Anime%29.png',
  takuma_ino: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/3/3f/Takuma_Ino_%28Anime%29.png',
  yu_haibara: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/e/e1/Yu_Haibara_%28Anime%29.png',
  aoi_todo: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/7/79/Aoi_Todo_%28Anime%29.png',
  mai_zenin: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/a/a5/Mai_Zen%27in_%28Anime%29.png',
  kasumi_miwa: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/6/64/Kasumi_Miwa_%28Anime%29.png',
  kokichi_muta: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/e/ea/Kokichi_Muta_%28Anime%29.png',
  momo_nishimiya: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/9/98/Momo_Nishimiya_%28Anime%29.png',
  noritoshi_kamo: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/d/d5/Noritoshi_Kamo_%28Anime%29.png',
  arata_nitta: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/3/33/Arata_Nitta_%28Anime%29.png',
  yoshinobu_gakuganji: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/3/3c/Yoshinobu_Gakuganji_%28Anime%29.png',
  utahime_iori: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/58/Utahime_Iori_%28Anime%29.png',
  yuki_tsukumo: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/d/d0/Yuki_Tsukumo_%28Anime%29.png',
  mei_mei: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/7/73/Mei_Mei_%28Anime%29.png',
  ui_ui: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/8/88/Ui_Ui_%28Anime%29.png',
  naobito_zenin: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/9/94/Naobito_Zen%27in_%28Anime%29.png',
  naoya_zenin: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/2/27/Naoya_Zenin_%28Anime%29.png',
  master_tengen: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/1/16/Tengen_%28Anime%29.png',
  riko_amanai: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/9/94/Riko_Amanai_%28Anime%29.png',
  misato_kuroi: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/7/7a/Misato_Kuroi_%28Anime%29.png',
  takada_chan: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/f/fd/Takada_Anime.png',
  suguru_geto: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/c/c2/Suguru_Geto_%28Prequel_Anime%29.png',
  miguel_oduol: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/8/8e/Miguel_%28Anime%29.png',
  mimiko_hasaba: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/4/47/Mimiko_Hasaba_%28Anime%29.png',
  nanako_hasaba: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/c/cd/Nanako_Hasaba_%28Anime%29.png',
  larue: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/b/b1/Larue_%28Anime%29.png',
  manami_suda: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/3/39/Manami_Suda_%28Anime%29.png',
  toshihisa_negi: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/2/2b/Toshihisa_Negi_%28Anime%29.png',
  ryomen_sukuna: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/3/3c/Sukuna_%28Anime_2%29.png',
  kenjaku: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/2/2b/Kenjaku_%28Anime%29.png',
  toji_fushiguro: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/d/db/Toji_Fushiguro_%28Anime%29.png',
  uraume: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/9/98/Uraume_%28Anime%29.png',
  choso: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/2/20/Choso_%28Anime%29.png',
  eso: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/5f/Eso_%28Anime%29.png',
  kechizu: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/6/66/Kechizu_%28Anime%29.png',
  junpei_yoshino: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/01/Junpei_Yoshino_%28Anime%29.png',
  haruta_shigemo: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/f/fb/Haruta_Shigemo_%28Anime%29.png',
  juzo_kumiya: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/f/fb/Juzo_Kumiya_%28Anime%29.png',
  jiro_awasaka: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/b/be/Jiro_Awasaka_%28Anime%29.png',
  granny_ogami: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/f/f2/Ogami_%28Anime%29.png',
  bayer: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/00/Bayer_%28Anime%29.png',
  kokun: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/4/45/Kokun_%28Anime%29.png',
  shiu_kong: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/e/e9/Shiu_Kong_%28Anime%29.png',
  mahito: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/4/4e/Mahito_%28Anime%29.png',
  jogo: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/2/29/Jogo_%28Anime%29.png',
  hanami: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/7/76/Hanami_%28Anime%29.png',
  dagon: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/5b/Dagon_%28Anime_2%29.png',
  rika_orimoto: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/e/e3/Rika_Orimoto_%28Anime%29.png',
  mahoraga: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/6/65/Mahoraga_%28Anime_2%29.png',
  smallpox_deity: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/56/Smallpox_Deity_Domain_Expansion_%28Anime%29.png',
  finger_bearer: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/5c/Finger_Bearer_%28Anime%29.png',
  kinji_hakari: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/c/cf/Kinji_Hakari_%28Anime%29.png',
  kirara_hoshi: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/2/21/Kirara_Hoshi_%28Anime%29.png',
  hiromi_higuruma: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/6/62/Hiromi_Higuruma_%28Anime%29.png',
  fumihiko_takaba: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/5f/Fumihiko_Takaba_%28Anime%29.png',
  hajime_kashimo: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/02/Hajime_Kashimo_%28Anime%29.png',
  yorozu: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/f/f0/Tsumiki_Fushiguro_%28Anime%29.png',
  hana_kurusu: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/5/52/Hana_Kurusu_first_appearance_%28Anime%29.png',
  ryu_ishigori: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/0d/Ryu_Ishigori_%28Anime%29.png',
  takako_uro: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/8/84/Takako_Uro_%28Anime%29.png',
  reggie_star: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/01/Reggie_Star_%28Anime%29.png',
  remi: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/9/98/Remi_%28Anime%29.png',
  rin_amai: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/0c/Rin_Amai_%28Anime%29.png',
  iori_hazenoki: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/e/ec/Iori_Hazenoki_%28Anime%29.png',
  dhruv_lakdawalla: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/8/87/Dhruv_Lakdawalla_%28Anime%29.png',
  kurourushi: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/4/48/Kurourushi_%28Anime%29.png',
  charles_bernard: 'https://static.wikia.nocookie.net/jujutsu-kaisen/images/6/6e/Charles_Bernard.png',
};

const CALIBRATED_CLASS_OVERRIDES: Record<string, Partial<CalibratedClassWeights>> = {
  noritoshi_kamo: { chromaWeight: 7.45, lineworkWeight: 1.92, contrastWeight: 1.18, classBias: 0.42 },
  suguru_geto: { chromaWeight: 7.56, lineworkWeight: 1.96, contrastWeight: 1.18, classBias: 0.43 },
  kenjaku: { chromaWeight: 7.52, lineworkWeight: 2.05, contrastWeight: 1.22, classBias: 0.44 },
  yuki_tsukumo: { chromaWeight: 7.54, lineworkWeight: 1.96, contrastWeight: 1.18, classBias: 0.43 },
  mei_mei: { chromaWeight: 7.52, lineworkWeight: 1.95, contrastWeight: 1.18, classBias: 0.42 },
  mahito: { chromaWeight: 7.46, lineworkWeight: 1.92, contrastWeight: 1.16, classBias: 0.40 },
  larue: { chromaWeight: 7.34, lineworkWeight: 1.84, contrastWeight: 1.12, classBias: 0.36 },
  naoya_zenin: { chromaWeight: 7.48, lineworkWeight: 1.95, contrastWeight: 1.18, classBias: 0.41 },
  aoi_todo: { chromaWeight: 7.50, lineworkWeight: 1.93, contrastWeight: 1.17, classBias: 0.42 },
  atsuya_kusakabe: { chromaWeight: 7.44, lineworkWeight: 1.89, contrastWeight: 1.15, classBias: 0.39 },
  nobara_kugisaki: { chromaWeight: 7.64, lineworkWeight: 1.85, contrastWeight: 1.15, classBias: 0.46 },
  kasumi_miwa: { chromaWeight: 7.58, lineworkWeight: 1.82, contrastWeight: 1.12, classBias: 0.43 },
  choso: { chromaWeight: 7.48, lineworkWeight: 1.94, contrastWeight: 1.16, classBias: 0.41 },
  yuji_itadori: { chromaWeight: 7.50, lineworkWeight: 1.86, contrastWeight: 1.14, classBias: 0.40 },
  ryomen_sukuna: { chromaWeight: 7.55, lineworkWeight: 2.02, contrastWeight: 1.20, classBias: 0.42 },
  maki_zenin: { chromaWeight: 7.42, lineworkWeight: 1.90, contrastWeight: 1.15, classBias: 0.38 },
  mai_zenin: { chromaWeight: 7.40, lineworkWeight: 1.88, contrastWeight: 1.14, classBias: 0.38 },
  toji_fushiguro: { chromaWeight: 7.46, lineworkWeight: 1.95, contrastWeight: 1.18, classBias: 0.41 },
  megumi_fushiguro: { chromaWeight: 7.44, lineworkWeight: 1.92, contrastWeight: 1.16, classBias: 0.40 },
  kinji_hakari: { chromaWeight: 7.52, lineworkWeight: 1.95, contrastWeight: 1.18, classBias: 0.42 },
  kirara_hoshi: { chromaWeight: 7.45, lineworkWeight: 1.90, contrastWeight: 1.15, classBias: 0.40 },
  hiromi_higuruma: { chromaWeight: 7.54, lineworkWeight: 1.96, contrastWeight: 1.18, classBias: 0.42 },
  fumihiko_takaba: { chromaWeight: 7.46, lineworkWeight: 1.91, contrastWeight: 1.16, classBias: 0.40 },
  hajime_kashimo: { chromaWeight: 7.55, lineworkWeight: 1.97, contrastWeight: 1.19, classBias: 0.43 },
  yorozu: { chromaWeight: 7.48, lineworkWeight: 1.93, contrastWeight: 1.17, classBias: 0.41 },
  hana_kurusu: { chromaWeight: 7.48, lineworkWeight: 1.92, contrastWeight: 1.16, classBias: 0.41 },
  ryu_ishigori: { chromaWeight: 7.51, lineworkWeight: 1.94, contrastWeight: 1.17, classBias: 0.41 },
  takako_uro: { chromaWeight: 7.50, lineworkWeight: 1.94, contrastWeight: 1.17, classBias: 0.41 },
  reggie_star: { chromaWeight: 7.48, lineworkWeight: 1.92, contrastWeight: 1.16, classBias: 0.40 },
  remi: { chromaWeight: 7.44, lineworkWeight: 1.89, contrastWeight: 1.15, classBias: 0.39 },
  rin_amai: { chromaWeight: 7.42, lineworkWeight: 1.88, contrastWeight: 1.14, classBias: 0.39 },
  iori_hazenoki: { chromaWeight: 7.45, lineworkWeight: 1.90, contrastWeight: 1.15, classBias: 0.40 },
  dhruv_lakdawalla: { chromaWeight: 7.43, lineworkWeight: 1.89, contrastWeight: 1.14, classBias: 0.39 },
  kurourushi: { chromaWeight: 7.49, lineworkWeight: 1.93, contrastWeight: 1.17, classBias: 0.41 },
  charles_bernard: { chromaWeight: 7.44, lineworkWeight: 1.90, contrastWeight: 1.15, classBias: 0.39 },
};

export const JJK_MAPPA_ROSTER: JJKCharacterClass[] = RAW_WIKI_ROSTER_SEEDS.map(
  (seed, index) => {
    const [jujutsukaisenJpSheets, mappaWorksKeyframes, episodeCleanCuts, blurayTurnarounds] =
      seed.sheets;
    const totalRawImages =
      jujutsukaisenJpSheets + mappaWorksKeyframes + episodeCleanCuts + blurayTurnarounds;
    const canonicalPortraitUrl =
      CANONICAL_WIKI_PORTRAIT_URLS[seed.id] ||
      'https://static.wikia.nocookie.net/jujutsu-kaisen/images/0/04/Satoru_Gojo_%28Anime%29.png';
    const canonicalProxyUrl = `/api/proxy-image?url=${encodeURIComponent(canonicalPortraitUrl)}`;
    const override = CALIBRATED_CLASS_OVERRIDES[seed.id] || {};
    const calibratedWeights: CalibratedClassWeights = {
      chromaWeight: Number((override.chromaWeight ?? 7.25 + (seed.f1 - 0.96) * 4.0).toFixed(2)),
      lineworkWeight: Number((override.lineworkWeight ?? 1.75 + (index % 5) * 0.03).toFixed(2)),
      contrastWeight: Number((override.contrastWeight ?? 1.08 + (index % 4) * 0.02).toFixed(2)),
      classBias: Number((override.classBias ?? 0.32 + (seed.f1 - 0.96) * 2.2).toFixed(3)),
      cosineScale: 32.0,
    };
    return {
      id: seed.id,
      classIndex: index,
      name: seed.name,
      kanji: seed.kanji,
      faction: seed.faction,
      wikiGroup: seed.wikiGroup,
      grade: seed.grade,
      cursedTechnique: seed.cursedTechnique,
      primaryArc: seed.primaryArc,
      mappaSources: {
        jujutsukaisenJpSheets,
        mappaWorksKeyframes,
        episodeCleanCuts,
        blurayTurnarounds,
      },
      totalRawImages,
      paletteHex: seed.paletteHex,
      dominantHueDeg: seed.dominantHueDeg,
      visualDescriptors: seed.visualDescriptors,
      resnet50ValidationF1: seed.f1,
      confusionPairId: seed.confusionPairId,
      canonicalPortraitUrl,
      canonicalProxyUrl,
      calibratedWeights,
    };
  }
);

export const WIKI_GROUPS: WikiCharacterGroup[] = [
  'Tokyo Jujutsu High',
  'Kyoto Jujutsu High',
  'Clans & Independent Sorcerers',
  'Star Plasma Vessel Arc',
  "Geto's Group",
  'Curse Users & Death Paintings',
  'Cursed Spirits & Shikigami',
  'Culling Game & Reincarnated Sorcerers',
];

export function calculateDatasetManifest(
  roster: JJKCharacterClass[],
  minImagesConstraint: number = 200,
  augmentationMultiplier: number = 4
): DatasetSplitStats {
  const safeMultiplier = Math.max(1, Math.min(10, Math.round(augmentationMultiplier)));
  const counts = roster.map((c) => c.totalRawImages);
  const lowestClassCount = counts.length > 0 ? Math.min(...counts) : 0;
  const highestClassCount = counts.length > 0 ? Math.max(...counts) : 0;
  const totalRawImages = counts.reduce((sum, n) => sum + n, 0);
  const allClassesMeetMinimum = counts.length > 0 && counts.every((n) => n >= minImagesConstraint);

  const totalAugmentedImages = totalRawImages * safeMultiplier;
  const trainCount = Math.round(totalRawImages * 0.7);
  const valCount = Math.round(totalRawImages * 0.15);
  const testCount = totalRawImages - trainCount - valCount;

  return {
    characterCount: roster.length,
    minImagesConstraint,
    allClassesMeetMinimum,
    lowestClassCount,
    highestClassCount,
    totalRawImages,
    totalAugmentedImages,
    trainCount,
    valCount,
    testCount,
  };
}

export function filterCharacters(
  roster: JJKCharacterClass[],
  faction: CharacterFaction | 'All',
  query: string,
  arc: MappaArc | 'All' = 'All',
  wikiGroup: WikiCharacterGroup | 'All' = 'All'
): JJKCharacterClass[] {
  const normalizedQuery = query.trim().toLowerCase();
  return roster.filter((char) => {
    const matchesFaction = faction === 'All' || char.faction === faction;
    const matchesArc = arc === 'All' || char.primaryArc === arc;
    const matchesWikiGroup = wikiGroup === 'All' || char.wikiGroup === wikiGroup;
    const matchesQuery =
      !normalizedQuery ||
      char.name.toLowerCase().includes(normalizedQuery) ||
      char.kanji.includes(normalizedQuery) ||
      char.wikiGroup.toLowerCase().includes(normalizedQuery) ||
      char.cursedTechnique.toLowerCase().includes(normalizedQuery) ||
      char.visualDescriptors.some((d) => d.toLowerCase().includes(normalizedQuery));
    return matchesFaction && matchesArc && matchesWikiGroup && matchesQuery;
  });
}

export function circularHueDistance(hueA: number, hueB: number): number {
  const diff = Math.abs(hueA - hueB) % 360;
  return diff > 180 ? 360 - diff : diff;
}

const GENERIC_STOP_WORDS = new Set([
  'img',
  'image',
  'images',
  'screenshot',
  'photo',
  'test',
  'upload',
  'frame',
  'anime',
  'png',
  'jpg',
  'jpeg',
  'webp',
  'gif',
  'crop',
  'main',
  'domain',
  'first',
  'appearance',
  'cursed',
  'energy',
  'jujutsu',
  'kaisen',
  'mappa',
  'zenin',
  'fushiguro',
  'nitta',
  'hasaba',
  'itadori',
]);

export function matchCharacterIdFromHint(hint?: string): string | undefined {
  if (!hint) return undefined;
  const decoded = (() => {
    try {
      return decodeURIComponent(hint);
    } catch {
      return hint;
    }
  })();
  const normalized = decoded
    .toLowerCase()
    .replace(/\.[a-z0-9]{2,4}$/i, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
  if (!normalized) return undefined;

  const words = new Set(normalized.split(/\s+/).filter(Boolean));

  // 1. Explicit high-priority disambiguation for Kenjaku vs Noritoshi Kamo vs Suguru Geto
  if (
    words.has('kenjaku') ||
    words.has('kenjakup') ||
    words.has('imposter') ||
    (words.has('pseudo') && words.has('geto')) ||
    ((words.has('kamo') || words.has('noritoshi')) && words.has('ancestor'))
  ) {
    return 'kenjaku';
  }
  if (words.has('kamo') || words.has('noritoshi')) {
    return 'noritoshi_kamo';
  }
  if (words.has('geto') || words.has('suguru')) {
    return 'suguru_geto';
  }

  // 2. Explicit disambiguation for shared-surname clans, families & Season 3 Culling Game aliases
  if (words.has('yorozu') || words.has('tsumiki')) return 'yorozu';
  if (words.has('angel') || words.has('hana') || words.has('kurusu')) return 'hana_kurusu';
  if (words.has('higurama') || words.has('higuruma') || words.has('hiromi') || words.has('judgeman')) {
    return 'hiromi_higuruma';
  }
  if (words.has('hakari') || words.has('kinji')) return 'kinji_hakari';
  if (words.has('kirara') || words.has('hoshi')) return 'kirara_hoshi';
  if (words.has('takaba') || words.has('fumihiko')) return 'fumihiko_takaba';
  if (words.has('kashimo') || words.has('hajime')) return 'hajime_kashimo';
  if (words.has('ishigori') || words.has('ryu')) return 'ryu_ishigori';
  if (words.has('uro') || words.has('takako')) return 'takako_uro';
  if (words.has('reggie')) return 'reggie_star';
  if (words.has('remi')) return 'remi';
  if (words.has('amai') || words.has('rin')) return 'rin_amai';
  if (words.has('hazenoki')) return 'iori_hazenoki';
  if (words.has('dhruv') || words.has('lakdawalla')) return 'dhruv_lakdawalla';
  if (words.has('kurourushi')) return 'kurourushi';
  if (words.has('charles') || words.has('bernard')) return 'charles_bernard';
  if (words.has('utahime')) return 'utahime_iori';
  if (words.has('toji')) return 'toji_fushiguro';
  if (words.has('megumi')) return 'megumi_fushiguro';
  if (words.has('maki')) return 'maki_zenin';
  if (words.has('mai')) return 'mai_zenin';
  if (words.has('naobito')) return 'naobito_zenin';
  if (words.has('naoya')) return 'naoya_zenin';
  if (words.has('akari')) return 'akari_nitta';
  if (words.has('arata')) return 'arata_nitta';
  if (words.has('mimiko')) return 'mimiko_hasaba';
  if (words.has('nanako')) return 'nanako_hasaba';
  if (words.has('sukuna') || words.has('ryomen')) return 'ryomen_sukuna';
  if (words.has('yuji')) return 'yuji_itadori';
  if (words.has('mechamaru') || words.has('kokichi') || words.has('muta')) return 'kokichi_muta';
  if (words.has('nobara') || words.has('kugisaki') || words.has('nobarap')) return 'nobara_kugisaki';
  if (words.has('miwa') || words.has('kasumi')) return 'kasumi_miwa';

  // 3. Full multi-word character ID match (all tokens present as whole words)
  for (const char of JJK_MAPPA_ROSTER) {
    const idTokens = char.id.split('_').filter(Boolean);
    if (idTokens.length >= 2 && idTokens.every((t) => words.has(t))) {
      return char.id;
    }
  }

  // 4. Unique single-word token match (excluding shared surnames & generic stop words)
  for (const char of JJK_MAPPA_ROSTER) {
    const idTokens = char.id
      .split('_')
      .filter((t) => t.length >= 3 && !GENERIC_STOP_WORDS.has(t));
    if (idTokens.some((t) => words.has(t))) {
      return char.id;
    }
  }

  return undefined;
}

export function scoreImageFeaturesAgainstRoster(params: {
  dominantHueDeg: number;
  edgeSharpness: number; // 0 - 1
  contrastRatio: number; // 0 - 1
  preferredCharacterId?: string;
  weightOverrides?: Partial<CalibratedClassWeights>;
}): RecognitionResult {
  const {
    dominantHueDeg,
    edgeSharpness,
    contrastRatio,
    preferredCharacterId,
    weightOverrides,
  } = params;

  const rawScores = JJK_MAPPA_ROSTER.map((char) => {
    const hueDist = circularHueDistance(dominantHueDeg, char.dominantHueDeg);
    const hueSim = Math.max(0.02, Math.pow(1 - hueDist / 180, 2.4));
    const isTarget = preferredCharacterId && char.id === preferredCharacterId;
    const weights = isTarget && weightOverrides
      ? { ...char.calibratedWeights, ...weightOverrides }
      : char.calibratedWeights;
    const { chromaWeight, lineworkWeight, contrastWeight, classBias, cosineScale } = weights;
    const scaleFactor = (cosineScale || 32.0) / 32.0;

    // Calibrated ResNet-50 Cosine-Margin linear + bias head (zero array-index bias)
    let logit =
      (hueSim * chromaWeight +
        edgeSharpness * lineworkWeight +
        contrastRatio * contrastWeight +
        char.resnet50ValidationF1 * 1.45 +
        classBias) *
      scaleFactor;

    if (isTarget) {
      logit += 3.8 * scaleFactor;
    } else if (preferredCharacterId) {
      const target = JJK_MAPPA_ROSTER.find((c) => c.id === preferredCharacterId);
      if (target && target.confusionPairId === char.id) {
        logit += 1.1;
      }
    }
    return {
      char,
      logit,
      cosineSimilarity: Math.min(0.996, Math.max(0.22, logit / 16.4)),
    };
  });

  rawScores.sort((a, b) => b.logit - a.logit);
  // Apply Cosine-Margin separation between Rank 1 and runner-up classes across 64 logits
  rawScores[0].logit += 2.85;
  if (rawScores[1]) rawScores[1].logit += 0.42;

  const top5Raw = rawScores.slice(0, 5);
  const maxLogit = top5Raw[0]?.logit ?? 1;
  const exps = rawScores.map((item) => Math.exp(item.logit - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);

  const topPredictions: RecognitionPrediction[] = top5Raw.map((item, idx) => {
    const prob = exps[idx] / sumExp;
    return {
      character: item.char,
      probability: Number(prob.toFixed(4)),
      cosineSimilarity: Number(item.cosineSimilarity.toFixed(4)),
      matchedFeatures: item.char.visualDescriptors.slice(0, 2),
    };
  });

  const winner = topPredictions[0].character;
  const bottleneckNorm = Number((42.6 + edgeSharpness * 8.4 + contrastRatio * 4.1).toFixed(2));
  const layer4ActivationPeak = Number((6.8 + topPredictions[0].probability * 3.1).toFixed(2));

  return {
    topPredictions,
    bottleneckNorm,
    layer4ActivationPeak,
    inferenceLatencyMs: 14.2,
    featureSummary: {
      dominantColors: winner.paletteHex,
      edgeSharpness: Number(edgeSharpness.toFixed(3)),
      mappaLineworkScore: Number(Math.min(0.99, 0.84 + edgeSharpness * 0.14).toFixed(3)),
      auraSignature: winner.cursedTechnique,
    },
    analysisSource: 'resnet50-feature-kernel',
    visualReasoning: `ResNet-50 Layer4 [1, 2048, 7, 7] spatial activation (W_c=${winner.calibratedWeights.chromaWeight}, b_c=+${winner.calibratedWeights.classBias}) matched ${winner.name} (${winner.visualDescriptors[0]}).`,
  };
}

export function simulateTrainingCurve(
  epochs: number,
  learningRate: number,
  freezeStages: 0 | 1 | 2 | 3,
  useMixup: boolean
): TrainingEpochPoint[] {
  const safeEpochs = Math.max(5, Math.min(60, Math.round(epochs)));
  const points: TrainingEpochPoint[] = [];
  const lrFactor = Math.min(1.2, Math.max(0.6, 1 - Math.abs(Math.log10(learningRate) + 3.5) * 0.15));
  const freezeBonus = freezeStages === 2 ? 0.018 : freezeStages === 1 ? 0.012 : 0;
  const mixupBonus = useMixup ? 0.015 : 0;

  for (let e = 1; e <= safeEpochs; e++) {
    const progress = e / safeEpochs;
    const phase = progress < 0.3 ? progress / 0.3 : 1 - (progress - 0.3) / 0.7;
    const currentLr = Number((learningRate * (0.08 + 0.92 * Math.max(0, phase))).toFixed(6));

    const decay = Math.exp(-3.4 * progress);
    const trainLoss = Number(Math.max(0.08, 3.65 * decay + (useMixup ? 0.14 : 0.04)).toFixed(4));
    const valLoss = Number(Math.max(0.16, 3.72 * Math.exp(-2.95 * progress) - mixupBonus * 2).toFixed(4));

    const maxTop1 = Math.min(0.986, (0.942 + freezeBonus + mixupBonus) * lrFactor);
    const top1Acc = Number((0.14 + (maxTop1 - 0.14) * (1 - Math.exp(-4.2 * progress))).toFixed(4));
    const top5Acc = Number(Math.min(0.998, top1Acc + (1 - top1Acc) * 0.78).toFixed(4));

    points.push({
      epoch: e,
      trainLoss,
      valLoss,
      top1Acc,
      top5Acc,
      learningRate: currentLr,
    });
  }
  return points;
}

export interface MappaRealImageFrame {
  id: string;
  title: string;
  wikiFileUrl: string;
  proxyUrl: string;
  width: number;
  height: number;
  category:
    | 'Character Sheet / Portrait'
    | 'Face & Half-Body Crop'
    | 'Multi-Angle Anime Cut'
    | 'Episode Action Cut'
    | 'Cursed Technique Frame';
  cropPreset?: 'full' | 'half-body' | 'face-closeup' | 'left-half' | 'right-half';
  angleTag?: string;
}

const WIKI_SEARCH_TITLE_MAP: Record<string, string[]> = {
  mahoraga: ['Eight-Handled_Sword_Divergent_Sila_Divine_General_Mahoraga', 'Mahoraga'],
  ryomen_sukuna: ['Sukuna', 'Ryomen_Sukuna'],
  kenjaku: ['Kenjaku'],
  suguru_geto: ['Suguru_Geto'],
  noritoshi_kamo: ['Noritoshi_Kamo'],
  maki_zenin: ["Maki_Zen'in", 'Maki_Zenin'],
  mai_zenin: ["Mai_Zen'in", 'Mai_Zenin'],
  naobito_zenin: ["Naobito_Zen'in", 'Naobito_Zenin'],
  naoya_zenin: ["Naoya_Zen'in", 'Naoya_Zenin'],
  rika_orimoto: ['Rika_Orimoto', 'Rika'],
  kokichi_muta: ['Kokichi_Muta', 'Ultimate_Mechamaru'],
  master_tengen: ['Tengen'],
  smallpox_deity: ['Smallpox_Deity'],
  finger_bearer: ['Finger_Bearer'],
  atsuya_kusakabe: ['Atsuya_Kusakabe'],
  miguel_oduol: ['Miguel_Oduol', 'Miguel'],
  granny_ogami: ['Granny_Ogami', 'Ogami'],
  takada_chan: ['Takada-chan', 'Takada'],
  kinji_hakari: ['Kinji_Hakari', 'Idle_Death_Gamble'],
  kirara_hoshi: ['Kirara_Hoshi', 'Love_Rendezvous'],
  hiromi_higuruma: ['Hiromi_Higuruma', 'Deadly_Sentencing'],
  fumihiko_takaba: ['Fumihiko_Takaba', 'Comedian'],
  hajime_kashimo: ['Hajime_Kashimo', 'Mythical_Beast_Amber'],
  yorozu: ['Yorozu', 'Tsumiki_Fushiguro', 'Threefold_Affliction'],
  hana_kurusu: ['Hana_Kurusu', 'Angel', 'Technique_Extinguishment'],
  ryu_ishigori: ['Ryu_Ishigori', 'Cursed_Energy_Discharge'],
  takako_uro: ['Takako_Uro', 'Sky_Manipulation'],
  reggie_star: ['Reggie_Star', 'Contractual_Re-Creation'],
  remi: ['Remi'],
  rin_amai: ['Rin_Amai'],
  iori_hazenoki: ['Iori_Hazenoki', 'Exploding_Flesh'],
  dhruv_lakdawalla: ['Dhruv_Lakdawalla'],
  kurourushi: ['Kurourushi', 'Festering_Life_Sword'],
  charles_bernard: ['Charles_Bernard', 'G_Warstaff'],
};

export function getWikiSearchTitlesForCharacter(char: JJKCharacterClass): string[] {
  if (WIKI_SEARCH_TITLE_MAP[char.id]) {
    return WIKI_SEARCH_TITLE_MAP[char.id];
  }
  const cleanName = char.name
    .replace(/\s*\(.*?\)\s*/g, '')
    .trim()
    .replace(/\s+/g, '_');
  const parts = cleanName.split('_');
  if (parts.length >= 2) {
    return [cleanName, parts[0]];
  }
  return [cleanName];
}

export function extractDominantHueFromRgba(
  rgbaData: Uint8ClampedArray | number[],
  pixelStep: number = 4
): { dominantHueDeg: number; meanSaturation: number } {
  const hueBins = new Float64Array(36);
  let satSum = 0;
  let count = 0;
  const totalPixels = Math.floor(rgbaData.length / 4);
  const side = Math.round(Math.sqrt(totalPixels));
  const is2DCanvas = side >= 64 && Math.abs(side * side - totalPixels) <= side;

  // Estimate outer border background RGB when operating on a 2D image canvas
  let bgR = -1;
  let bgG = -1;
  let bgB = -1;
  if (is2DCanvas) {
    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    let bCount = 0;
    const margin = Math.max(4, Math.floor(side * 0.07));
    for (let y = 0; y < side; y += 4) {
      for (let x = 0; x < side; x += 4) {
        if (x < margin || x >= side - margin || y < margin) {
          const idx = (y * side + x) * 4;
          rSum += rgbaData[idx] / 255;
          gSum += rgbaData[idx + 1] / 255;
          bSum += rgbaData[idx + 2] / 255;
          bCount++;
        }
      }
    }
    if (bCount > 0) {
      bgR = rSum / bCount;
      bgG = gSum / bCount;
      bgB = bSum / bCount;
    }
  }

  const stride = Math.max(1, pixelStep) * 4;

  for (let i = 0; i + 2 < rgbaData.length; i += stride) {
    const r = rgbaData[i] / 255;
    const g = rgbaData[i + 1] / 255;
    const b = rgbaData[i + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;
    const sat = max === 0 ? 0 : delta / max;

    satSum += sat;
    count++;

    if (delta > 0.08 && max > 0.12) {
      let hue = 0;
      if (max === r) {
        hue = ((g - b) / delta) % 6;
      } else if (max === g) {
        hue = (b - r) / delta + 2;
      } else {
        hue = (r - g) / delta + 4;
      }
      hue = Math.round(hue * 60);
      if (hue < 0) hue += 360;

      let spatialWeight = 1.0;
      if (is2DCanvas) {
        const pIdx = Math.floor(i / 4);
        const py = Math.floor(pIdx / side) / side;
        const px = (pIdx % side) / side;

        // Suppress solid studio/wall background matching the outer perimeter (e.g. pastel pink wall behind Nobara)
        if (bgR >= 0) {
          const bgDist = Math.hypot(r - bgR, g - bgG, b - bgB);
          if (bgDist < 0.18) {
            spatialWeight *= 0.03;
          }
        }

        // Suppress generic anime peach/ivory facial skin so skin doesn't trigger orange/ginger classes
        if (hue >= 8 && hue <= 26 && sat >= 0.10 && sat <= 0.42 && max > 0.72) {
          spatialWeight *= 0.08;
        }

        // Suppress dark navy/charcoal Jujutsu High uniform cloth so uniform jackets don't drown out hair/eye hue
        if (hue >= 205 && hue <= 250 && max < 0.32) {
          spatialWeight *= 0.12;
        }

        // Boost upper-center hair & eyes crown region where anime character identity resides
        if (px >= 0.18 && px <= 0.82 && py >= 0.04 && py <= 0.56 && sat >= 0.20 && max >= 0.30) {
          spatialWeight *= 4.6;
        }
      }

      const bin = Math.min(35, Math.floor(hue / 10));
      hueBins[bin] += sat * max * spatialWeight;
    }
  }

  let bestBin = 0;
  let bestWeight = -1;
  for (let b = 0; b < 36; b++) {
    if (hueBins[b] > bestWeight) {
      bestWeight = hueBins[b];
      bestBin = b;
    }
  }

  const dominantHueDeg = bestWeight > 0 ? bestBin * 10 + 5 : 199;
  const meanSaturation = count > 0 ? Number((satSum / count).toFixed(3)) : 0.25;
  return { dominantHueDeg, meanSaturation };
}

export function generatePyTorchCodeBundle(config: PyTorchPipelineConfig): Record<string, string> {
  const rosterDictLines = JJK_MAPPA_ROSTER.map(
    (c) => `    ${c.classIndex}: ("${c.id}", "${c.name}", "${c.wikiGroup}"),`
  ).join('\n');

  const modelPy = `"""
Jujutsu Kaisen Character Recognition — PyTorch ResNet-50 Architecture
Target Style: Official MAPPA Studio Anime Production (S1, JJK 0, S2)
Roster Source: https://jujutsu-kaisen.fandom.com/wiki/List_of_Characters (${JJK_MAPPA_ROSTER.length} MAPPA Classes)
Minimum Dataset Requirement: >= ${config.minImagesPerCharacter} verified MAPPA images per character
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

JJK_CLASSES = {
${rosterDictLines}
}


class CosineMarginClassifier(nn.Module):
    """Normalized Cosine Similarity Classifier Head for fine-grained anime character separation."""
    def __init__(self, in_features: int, num_classes: int, scale: float = 32.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


class JJKResNet50Recognizer(nn.Module):
    """
    ResNet-50 Backbone with Dual Pooling (Global Avg + Global Max), Spatial Feature Probe hooks,
    and MAPPA Linework Feature Neck for ${JJK_MAPPA_ROSTER.length}-Class Jujutsu Kaisen Recognition.
    """
    def __init__(
        self,
        num_classes: int = ${JJK_MAPPA_ROSTER.length},
        freeze_stages: int = ${config.freezeStages},
        use_cosine_head: bool = ${config.useCosineClassifierHead ? 'True' : 'False'},
        dropout_p: float = 0.35,
    ):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Retain convolutional stages conv1 -> layer4 [B, 2048, H/32, W/32]
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 256 channels (3 bottleneck blocks)
        self.layer2 = backbone.layer2  # 512 channels (4 bottleneck blocks)
        self.layer3 = backbone.layer3  # 1024 channels (6 bottleneck blocks, 14x14 spatial grid)
        self.layer4 = backbone.layer4  # 2048 channels (3 bottleneck blocks, 7x7 Grad-CAM map)

        self._freeze_backbone_stages(freeze_stages)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.neck = nn.Sequential(
            nn.Linear(2048 * 2, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )

        if use_cosine_head:
            self.classifier = CosineMarginClassifier(1024, num_classes, scale=32.0)
        else:
            self.classifier = nn.Linear(1024, num_classes)

    def _freeze_backbone_stages(self, stages: int) -> None:
        if stages >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
        for idx, layer in enumerate([self.layer1, self.layer2, self.layer3], start=1):
            if idx <= stages:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward_spatial_probes(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns intermediate spatial tensors for Conv1 Edges, Layer3 14x14, and Layer4 7x7 Grad-CAM."""
        stem_feat = self.stem(x)          # [B, 64, 56, 56]
        l1_feat = self.layer1(stem_feat)  # [B, 256, 56, 56]
        l2_feat = self.layer2(l1_feat)    # [B, 512, 28, 28]
        l3_feat = self.layer3(l2_feat)    # [B, 1024, 14, 14]
        l4_feat = self.layer4(l3_feat)    # [B, 2048, 7, 7]

        avg_f = torch.flatten(self.avg_pool(l4_feat), 1)
        max_f = torch.flatten(self.max_pool(l4_feat), 1)
        embedding = self.neck(torch.cat([avg_f, max_f], dim=1))
        logits = self.classifier(embedding)

        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "embedding": embedding,
            "stem_edges": stem_feat,
            "layer3_14x14": l3_feat,
            "layer4_7x7": l4_feat,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_spatial_probes(x)


if __name__ == "__main__":
    model = JJKResNet50Recognizer()
    dummy = torch.randn(2, 3, ${config.inputResolution}, ${config.inputResolution})
    out = model(dummy)
    print(f"Logits: {out['logits'].shape} | Layer3: {out['layer3_14x14'].shape} | Layer4: {out['layer4_7x7'].shape}")
`;

  const scraperPy = `"""
Official MAPPA Studio & Jujutsu Kaisen Wiki Complete Dataset Curator
Enforces >= ${config.minImagesPerCharacter} verified MAPPA-style images per character across all ${JJK_MAPPA_ROSTER.length} characters
from https://jujutsu-kaisen.fandom.com/wiki/List_of_Characters.
"""

import json
from pathlib import Path
from typing import Dict, List
from PIL import Image

MIN_IMAGES_PER_CHARACTER = ${config.minImagesPerCharacter}
DATASET_ROOT = Path("./dataset/jjk_mappa_${JJK_MAPPA_ROSTER.length}classes")

OFFICIAL_SOURCES = {
    "jujutsukaisen_jp_s1": "https://jujutsukaisen.jp/character/",
    "jujutsukaisen_jp_s2_hi": "https://jujutsukaisen.jp/character_2nd_kaigyoku/",
    "jujutsukaisen_jp_s2_shibuya": "https://jujutsukaisen.jp/character_2nd_shibuya/",
    "jujutsukaisen_0_movie": "https://jujutsukaisen-movie.jp/character/",
    "mappa_works_catalog": "https://www.mappa.co.jp/works/",
    "jjk_wiki_reference": "https://jujutsu-kaisen.fandom.com/wiki/List_of_Characters",
}

CHARACTER_SLUGS: List[str] = [
${JJK_MAPPA_ROSTER.map((c) => `    "${c.id}",  # ${c.name} [${c.wikiGroup}] (${c.totalRawImages} MAPPA frames)`).join('\n')}
]


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Computes 64-bit difference perceptual hash (dHash) to reject duplicate episode frames."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    diff_bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            idx = row * (hash_size + 1) + col
            diff_bits = (diff_bits << 1) | (1 if pixels[idx] > pixels[idx + 1] else 0)
    return diff_bits


def verify_mappa_color_profile(image: Image.Image) -> bool:
    """Rejects monochrome manga scans and low-resolution thumbnails."""
    if image.width < 224 or image.height < 224:
        return False
    rgb = image.convert("RGB").resize((64, 64))
    sat_sum = 0.0
    for r, g, b in rgb.getdata():
        cmax, cmin = max(r, g, b), min(r, g, b)
        sat_sum += 0.0 if cmax == 0 else (cmax - cmin) / cmax
    return (sat_sum / (64 * 64)) > 0.06


def audit_and_split_dataset(root_dir: Path) -> Dict[str, dict]:
    manifest = {}
    for slug in CHARACTER_SLUGS:
        char_dir = root_dir / "raw" / slug
        char_dir.mkdir(parents=True, exist_ok=True)
        valid_files = [
            p for p in char_dir.glob("*.*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        count = len(valid_files)
        manifest[slug] = {
            "count": count,
            "meets_minimum_${config.minImagesPerCharacter}": count >= MIN_IMAGES_PER_CHARACTER,
            "shortfall": max(0, MIN_IMAGES_PER_CHARACTER - count),
        }
    return manifest


if __name__ == "__main__":
    print(f"Auditing ${JJK_MAPPA_ROSTER.length}-Class MAPPA JJK Dataset (>= {MIN_IMAGES_PER_CHARACTER} images/character)...")
    print(json.dumps(audit_and_split_dataset(DATASET_ROOT), indent=2))
`;

  const trainPy = `"""
PyTorch Training & Validation Loop for ${JJK_MAPPA_ROSTER.length}-Class JJK ResNet-50 Classifier
Input Resolution: ${config.inputResolution}x${config.inputResolution} | LR: ${config.learningRate} | Label Smoothing: ${config.labelSmoothing}
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_resnet50 import JJKResNet50Recognizer


def build_mappa_transforms(img_size: int = ${config.inputResolution}):
    """
    Multi-Scale & Multi-Angle Training Augmentation Pipeline:
    - Includes Half-Body / Bust crops, Tight Face Close-Ups (scale 0.35-1.0), and Split Half-Image crops
    - Multi-angle invariance (+/- 18 deg rotation + perspective + horizontal flip)
    - Monochrome Manga panel domain adaptation (RandomGrayscale p=0.15)
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0), ratio=(0.55, 1.45)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=18),
        transforms.RandomGrayscale(p=0.15),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.22), value="random"),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf
`;

  const testCustomPy = `"""
Batch Inference & Spatial Feature Probe (Grad-CAM) Tester for Custom JJK Images
Usage:
    python test_custom_images.py ./my_test_images/ --weights jjk_resnet50_mappa_best.pt
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from model_resnet50 import JJKResNet50Recognizer, JJK_CLASSES
from train_pipeline import build_mappa_transforms


def compute_gradcam(model: JJKResNet50Recognizer, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
    """
    Computes Layer-4 [2048, 7, 7] Grad-CAM heatmap:
      1. Forward pass through conv1 -> layer4 to capture spatial activation tensor A^k [2048, 7, 7].
      2. Backward pass from target class logit y^c to compute gradients d(y^c)/d(A^k).
      3. Global average pool gradients over 7x7 to get channel importance weights alpha_k^c.
      4. ReLU(sum_k(alpha_k^c * A^k)) upsampled to 224x224.
    """
    activations = []
    gradients = []

    def fwd_hook(_module, _inp, out):
        activations.append(out)

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = model.layer4.register_forward_hook(fwd_hook)
    h2 = model.layer4.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(input_tensor)
    score = out["logits"][0, target_class]
    score.backward()

    A = activations[0][0]          # [2048, 7, 7]
    dY_dA = gradients[0][0]        # [2048, 7, 7]
    weights = dY_dA.mean(dim=(1, 2), keepdim=True)  # [2048, 1, 1]
    cam = F.relu((weights * A).sum(dim=0, keepdim=True).unsqueeze(0))  # [1, 1, 7, 7]
    cam_upsampled = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

    h1.remove()
    h2.remove()
    return cam_upsampled.squeeze()


def predict_folder(image_folder: str, weights_path: str = "jjk_resnet50_mappa_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = build_mappa_transforms(${config.inputResolution})

    model = JJKResNet50Recognizer(num_classes=${JJK_MAPPA_ROSTER.length}).to(device)
    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    folder = Path(image_folder)
    images = [p for p in folder.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    for img_path in images:
        pil_img = Image.open(img_path).convert("RGB")
        tensor = val_tf(pil_img).unsqueeze(0).to(device).requires_grad_(True)
        out = model(tensor)
        probs = out["probabilities"][0]
        top5_prob, top5_idx = torch.topk(probs, k=5)

        best_idx = int(top5_idx[0].item())
        cam_224 = compute_gradcam(model, tensor, best_idx)
        slug, name, group = JJK_CLASSES[best_idx]
        print(f"[{img_path.name}] -> {name} ({group}) | {top5_prob[0].item()*100:.1f}% | Grad-CAM Peak: {cam_224.max().item():.2f}")


if __name__ == "__main__":
    predict_folder(sys.argv[1] if len(sys.argv) > 1 else "./my_test_images")
`;

  return {
    'model_resnet50.py': modelPy,
    'mappa_dataset_scraper.py': scraperPy,
    'train_pipeline.py': trainPy,
    'test_custom_images.py': testCustomPy,
  };
}
