import React, { useEffect, useState } from 'react';
import {
  CharacterFaction,
  JJK_MAPPA_ROSTER,
  JJKCharacterClass,
  MappaArc,
  MappaRealImageFrame,
  WIKI_GROUPS,
  WikiCharacterGroup,
  calculateDatasetManifest,
  filterCharacters,
} from '../utils/jjkClassifierLogic';
import {
  Search,
  CheckCircle2,
  AlertTriangle,
  Image as ImageIcon,
  Download,
  ExternalLink,
} from 'lucide-react';

interface DatasetRegistryProps {
  onInspectCharacterInWorkbench: (char: JJKCharacterClass) => void;
}

export const DatasetRegistry: React.FC<DatasetRegistryProps> = ({
  onInspectCharacterInWorkbench,
}) => {
  const [selectedFaction, setSelectedFaction] = useState<CharacterFaction | 'All'>('All');
  const [selectedWikiGroup, setSelectedWikiGroup] = useState<WikiCharacterGroup | 'All'>('All');
  const [selectedArc, setSelectedArc] = useState<MappaArc | 'All'>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [minThreshold] = useState<number>(200);
  const [augMultiplier] = useState<number>(4);

  // Active character whose real MAPPA anime images from jujutsu-kaisen.fandom.com are shown
  const [galleryCharacter, setGalleryCharacter] = useState<JJKCharacterClass>(
    () => JJK_MAPPA_ROSTER.find((c) => c.id === 'nobara_kugisaki') || JJK_MAPPA_ROSTER[0]
  );
  const [galleryFrames, setGalleryFrames] = useState<MappaRealImageFrame[]>([]);
  const [isLoadingGallery, setIsLoadingGallery] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    setIsLoadingGallery(true);
    fetch(`/api/character-images/${encodeURIComponent(galleryCharacter.id)}`)
      .then((res) => (res.ok ? res.json() : { images: [] }))
      .then((data) => {
        if (!cancelled) {
          setGalleryFrames(Array.isArray(data.images) ? data.images : []);
          setIsLoadingGallery(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setGalleryFrames([]);
          setIsLoadingGallery(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [galleryCharacter.id]);

  const handleDownloadGalleryJson = () => {
    const payload = {
      characterId: galleryCharacter.id,
      name: galleryCharacter.name,
      kanji: galleryCharacter.kanji,
      wikiGroup: galleryCharacter.wikiGroup,
      cursedTechnique: galleryCharacter.cursedTechnique,
      totalCuratedFrames: galleryCharacter.totalRawImages,
      mappaSourceBreakdown: galleryCharacter.mappaSources,
      officialWikiMappaAnimeFrames: galleryFrames.map((f) => ({
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
    a.download = `${galleryCharacter.id}_mappa_images.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const filteredCharacters = filterCharacters(
    JJK_MAPPA_ROSTER,
    selectedFaction,
    searchQuery,
    selectedArc,
    selectedWikiGroup
  );

  const manifestStats = calculateDatasetManifest(
    filteredCharacters,
    minThreshold,
    augMultiplier
  );

  const factions: (CharacterFaction | 'All')[] = [
    'All',
    'Jujutsu Sorcerer',
    'Cursed Spirit',
    'Curse User / Vessel',
  ];

  const arcs: (MappaArc | 'All')[] = [
    'All',
    'Season 1 ( Cour 1 & 2 )',
    'Jujutsu Kaisen 0 (Movie)',
    'Season 2: Hidden Inventory',
    'Season 2: Shibuya Incident',
    'Season 3: Culling Game',
  ];

  return (
    <section aria-labelledby="dataset-heading" className="space-y-8">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-6 border-b border-slate-800">
        <div className="space-y-2 max-w-3xl">
          <p className="text-xs font-mono text-cyan-400 tracking-wide">
            JUJUTSU KAISEN WIKI + OFFICIAL MAPPA STUDIO MANIFEST · LIVE MULTI-IMAGE GALLERY
          </p>
          <h1
            id="dataset-heading"
            className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100"
          >
            MAPPA Art Style Training Dataset &amp; Reference Image Browser ({JJK_MAPPA_ROSTER.length} Wiki Classes)
          </h1>
          <p className="text-sm text-slate-400 leading-relaxed">
            Browse the real MAPPA Studio anime reference frames (`(Anime).png`, character design
            sheets, Cursed Technique keyframes, half-images, and episode cuts) retrieved directly from the
            Jujutsu Kaisen Wiki for every character in the {JJK_MAPPA_ROSTER.length}-class roster (including Season 3 Culling Game &amp; Reincarnated Sorcerers).
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          {manifestStats.allClassesMeetMinimum ? (
            <span className="text-emerald-400 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" aria-hidden="true" />
              All {manifestStats.characterCount} classes pass ≥{minThreshold} images check (Min:{' '}
              {manifestStats.lowestClassCount})
            </span>
          ) : (
            <span className="text-amber-400 flex items-center gap-1.5">
              <AlertTriangle className="w-4 h-4" aria-hidden="true" />
              Threshold {minThreshold} exceeds lowest class ({manifestStats.lowestClassCount} raw frames)
            </span>
          )}
        </div>
      </div>

      {/* Live Multi-Image MAPPA Reference Gallery Viewer */}
      <div
        data-testid="dataset-live-image-gallery"
        className="p-5 bg-[#111827] border border-cyan-500/40 rounded-xl space-y-4"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-slate-800">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm font-bold text-slate-100">
              <ImageIcon className="w-4 h-4 text-cyan-400" aria-hidden="true" />
              <span>
                Real MAPPA Studio Reference Images: {galleryCharacter.name} ({galleryCharacter.kanji})
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Showing {galleryFrames.length} official `(Anime)` frames from{' '}
              <span className="font-mono text-slate-300">jujutsu-kaisen.fandom.com</span> · Click
              any character&apos;s &ldquo;View MAPPA Images&rdquo; button in the table below to
              inspect its frames.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <select
              aria-label="Select character to view real MAPPA reference images"
              value={galleryCharacter.id}
              onChange={(e) => {
                const found = JJK_MAPPA_ROSTER.find((c) => c.id === e.target.value);
                if (found) setGalleryCharacter(found);
              }}
              className="px-3 py-1.5 text-xs bg-slate-950 border border-slate-700 rounded-lg text-slate-100"
            >
              {JJK_MAPPA_ROSTER.map((c) => (
                <option key={c.id} value={c.id}>
                  #{String(c.classIndex).padStart(2, '0')} — {c.name}
                </option>
              ))}
            </select>

            <button
              type="button"
              onClick={handleDownloadGalleryJson}
              className="px-3 py-1.5 text-xs font-medium text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" aria-hidden="true" />
              <span>Download Image URLs (JSON)</span>
            </button>
          </div>
        </div>

        {isLoadingGallery ? (
          <div className="py-8 text-center text-xs font-mono text-slate-400">
            Fetching official MAPPA Studio anime frames for {galleryCharacter.name}...
          </div>
        ) : galleryFrames.length === 0 ? (
          <div className="py-6 text-center text-xs text-slate-400">
            No cached Wiki frames returned. Use the PyTorch scraper script (`mappa_dataset_scraper.py`) to pull the full {galleryCharacter.totalRawImages}-frame manifest.
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 max-h-[380px] overflow-y-auto pr-1">
            {galleryFrames.map((frame) => {
              const cropClass =
                frame.cropPreset === 'face-closeup'
                  ? 'scale-[1.7] origin-top'
                  : frame.cropPreset === 'half-body'
                    ? 'scale-[1.3] origin-top'
                    : frame.cropPreset === 'left-half'
                      ? 'scale-[1.4] origin-top-left'
                      : frame.cropPreset === 'right-half'
                        ? 'scale-[1.4] origin-top-right'
                        : '';
              return (
                <div
                  key={frame.id}
                  className="flex flex-col bg-slate-950 border border-slate-800 rounded-lg overflow-hidden hover:border-cyan-500/60 transition-colors"
                >
                  <div className="relative aspect-video w-full bg-slate-900 overflow-hidden">
                    <img
                      src={frame.proxyUrl}
                      alt={frame.title}
                      loading="lazy"
                      className={`w-full h-full object-cover object-top ${cropClass}`}
                    />
                    {frame.angleTag && (
                      <span className="absolute bottom-1 right-1 bg-slate-950/85 text-[9px] font-mono text-cyan-300 px-1.5 py-0.5 rounded">
                        {frame.angleTag}
                      </span>
                    )}
                  </div>
                  <div className="p-2.5 flex-1 flex flex-col justify-between space-y-1.5">
                    <div>
                      <div
                        className="text-[11px] font-medium text-slate-200 line-clamp-1"
                        title={frame.title}
                      >
                        {frame.title}
                      </div>
                      <div className="text-[10px] font-mono text-slate-400 tabular-nums">
                        {frame.category} · {frame.width}×{frame.height}
                      </div>
                    </div>
                    <a
                      href={frame.proxyUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] font-mono text-cyan-400 hover:text-cyan-300"
                    >
                      <span>Open Full Frame</span>
                      <ExternalLink className="w-3 h-3" aria-hidden="true" />
                    </a>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 4-Column Dataset Telemetry Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 bg-[#111827] border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">VERIFIED RAW MAPPA FRAMES</div>
          <div
            data-testid="total-raw-images"
            className="text-2xl font-mono font-bold text-slate-100 tabular-nums mt-1"
          >
            {manifestStats.totalRawImages.toLocaleString()}
          </div>
          <div className="text-xs text-slate-400 mt-1 font-mono tabular-nums">
            Min {manifestStats.lowestClassCount} · Max {manifestStats.highestClassCount} / class
          </div>
        </div>

        <div className="p-4 bg-[#111827] border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">AUGMENTED EPOCH CORPUS ({augMultiplier}×)</div>
          <div
            data-testid="total-augmented-images"
            className="text-2xl font-mono font-bold text-cyan-400 tabular-nums mt-1"
          >
            {manifestStats.totalAugmentedImages.toLocaleString()}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Hue-locked ±2% · RandomResizedCrop(224)
          </div>
        </div>

        <div className="p-4 bg-[#111827] border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">STRATIFIED SPLIT (70 / 15 / 15)</div>
          <div className="text-lg font-mono font-bold text-slate-100 tabular-nums mt-1.5">
            {manifestStats.trainCount} / {manifestStats.valCount} / {manifestStats.testCount}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Train / Validation / Holdout Test
          </div>
        </div>

        <div className="p-4 bg-[#111827] border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">WIKI ROSTER COVERAGE</div>
          <div className="text-2xl font-mono font-bold text-emerald-400 tabular-nums mt-1">
            {manifestStats.characterCount} / {JJK_MAPPA_ROSTER.length}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {WIKI_GROUPS.length} Wiki Categories · 64-bit dHash
          </div>
        </div>
      </div>

      {/* Filter & Threshold Control Bar */}
      <div className="p-5 bg-[#111827] border border-slate-800 rounded-xl space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
          {/* Faction Segmented Filter */}
          <div className="md:col-span-4 space-y-1.5">
            <span className="block text-xs font-medium text-slate-400">
              Filter by Character Faction
            </span>
            <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-950 border border-slate-800 rounded-lg">
              {factions.map((faction) => (
                <button
                  key={faction}
                  type="button"
                  onClick={() => setSelectedFaction(faction)}
                  className={`px-2 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    selectedFaction === faction
                      ? 'bg-cyan-400 text-slate-950 font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {faction === 'All' ? `All (${JJK_MAPPA_ROSTER.length})` : faction}
                </button>
              ))}
            </div>
          </div>

          {/* Wiki Category Filter */}
          <div className="md:col-span-3 space-y-1.5">
            <label htmlFor="wiki-category-filter" className="block text-xs font-medium text-slate-400">
              JJK Wiki Category
            </label>
            <select
              id="wiki-category-filter"
              value={selectedWikiGroup}
              onChange={(e) => setSelectedWikiGroup(e.target.value as WikiCharacterGroup | 'All')}
              className="w-full px-2.5 py-1.5 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-400"
            >
              <option value="All">All 7 Wiki Categories</option>
              {WIKI_GROUPS.map((grp) => (
                <option key={grp} value={grp}>
                  {grp}
                </option>
              ))}
            </select>
          </div>

          {/* Search Input */}
          <div className="md:col-span-3 space-y-1.5">
            <label htmlFor="character-search" className="block text-xs font-medium text-slate-400">
              Search Character or Technique
            </label>
            <div className="relative">
              <Search
                className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2"
                aria-hidden="true"
              />
              <input
                id="character-search"
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="e.g. Nobara, Miwa, Gojo, Mahoraga..."
                className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400"
              />
            </div>
          </div>

          {/* Arc Filter */}
          <div className="md:col-span-2 space-y-1.5">
            <label htmlFor="arc-filter" className="block text-xs font-medium text-slate-400">
              MAPPA Production Arc
            </label>
            <select
              id="arc-filter"
              value={selectedArc}
              onChange={(e) => setSelectedArc(e.target.value as MappaArc | 'All')}
              className="w-full px-2.5 py-1.5 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-400"
            >
              {arcs.map((arc) => (
                <option key={arc} value={arc}>
                  {arc === 'All' ? 'All MAPPA Arcs' : arc}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* High-Density Character Dataset Table */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] font-mono text-slate-400 bg-slate-950/60">
                <th className="py-3 px-4">ID</th>
                <th className="py-3 px-4">CHARACTER &amp; KANJI</th>
                <th className="py-3 px-4">WIKI GROUP &amp; GRADE</th>
                <th className="py-3 px-4">PRIMARY MAPPA ARC</th>
                <th className="py-3 px-4 text-right">DESIGN SHEETS</th>
                <th className="py-3 px-4 text-right">EPISODE CUTS</th>
                <th className="py-3 px-4 text-right">TOTAL FRAMES (≥200)</th>
                <th className="py-3 px-4 text-right">VAL F1</th>
                <th className="py-3 px-4 text-right">ACTIONS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 text-xs">
              {filteredCharacters.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-10 text-center text-slate-400">
                    No characters match "{searchQuery}".{' '}
                    <button
                      type="button"
                      onClick={() => {
                        setSearchQuery('');
                        setSelectedFaction('All');
                        setSelectedWikiGroup('All');
                        setSelectedArc('All');
                      }}
                      className="text-cyan-400 underline ml-2"
                    >
                      Reset Filters
                    </button>
                  </td>
                </tr>
              ) : (
                filteredCharacters.map((char) => {
                  const isGalleryActive = galleryCharacter.id === char.id;
                  return (
                    <tr
                      key={char.id}
                      className="hover:bg-slate-900/70 transition-colors"
                    >
                      <td className="py-3 px-4 font-mono text-slate-400 tabular-nums">
                        #{String(char.classIndex).padStart(2, '0')}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <img
                            src={char.canonicalProxyUrl}
                            alt={char.name}
                            loading="lazy"
                            className="w-9 h-9 rounded-md object-cover object-top bg-slate-900 border border-slate-700 shrink-0"
                          />
                          <div>
                            <div className="font-semibold text-slate-100">{char.name}</div>
                            <div className="text-[11px] text-slate-400">
                              {char.kanji} · {char.cursedTechnique}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-slate-300">
                        <div className="text-cyan-400 font-medium">{char.wikiGroup}</div>
                        <div className="text-[11px] text-slate-400">{char.grade}</div>
                      </td>
                      <td className="py-3 px-4 text-slate-300 whitespace-nowrap">
                        {char.primaryArc}
                      </td>
                      <td className="py-3 px-4 text-right font-mono text-slate-300 tabular-nums">
                        {char.mappaSources.jujutsukaisenJpSheets +
                          char.mappaSources.mappaWorksKeyframes +
                          char.mappaSources.blurayTurnarounds}
                      </td>
                      <td className="py-3 px-4 text-right font-mono text-slate-300 tabular-nums">
                        {char.mappaSources.episodeCleanCuts}
                      </td>
                      <td className="py-3 px-4 text-right font-mono font-semibold text-cyan-400 tabular-nums">
                        {char.totalRawImages} imgs
                      </td>
                      <td className="py-3 px-4 text-right font-mono text-emerald-400 tabular-nums">
                        {(char.resnet50ValidationF1 * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            type="button"
                            onClick={() => {
                              setGalleryCharacter(char);
                              window.scrollTo({ top: 0, behavior: 'smooth' });
                            }}
                            className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors whitespace-nowrap ${
                              isGalleryActive
                                ? 'bg-cyan-400 text-slate-950 border-cyan-400 font-semibold'
                                : 'text-slate-200 bg-slate-900 border-slate-700 hover:bg-slate-800'
                            }`}
                          >
                            View MAPPA Images
                          </button>
                          <button
                            type="button"
                            onClick={() => onInspectCharacterInWorkbench(char)}
                            className="px-2.5 py-1 text-xs font-medium text-cyan-300 bg-cyan-950/50 border border-cyan-800/70 rounded hover:bg-cyan-900/60 transition-colors whitespace-nowrap"
                          >
                            Test in Workbench
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};
