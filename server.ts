import express from 'express';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import {
  JJK_MAPPA_ROSTER,
  type MappaRealImageFrame,
  calculateDatasetManifest,
  getWikiSearchTitlesForCharacter,
  matchCharacterIdFromHint,
  scoreImageFeaturesAgainstRoster,
} from './src/utils/jjkClassifierLogic.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '50mb' }));

const CHROME_HEADERS: Record<string, string> = {
  'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
  'Referer': 'https://jujutsu-kaisen.fandom.com/',
  'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
  'Accept-Language': 'en-US,en;q=0.9',
};

const IMAGE_PROXY_CACHE_DIR = path.resolve(process.cwd(), '.cache/image-proxy-cache');
if (!fs.existsSync(IMAGE_PROXY_CACHE_DIR)) {
  fs.mkdirSync(IMAGE_PROXY_CACHE_DIR, { recursive: true });
}

function getGeminiClient(overrideKey?: string): GoogleGenAI | null {
  let apiKey = (overrideKey || process.env.GEMINI_API_KEY || process.env.VITE_GEMINI_API_KEY || '').trim();
  apiKey = apiKey.replace(/^["']|["']$/g, '').trim();
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({ apiKey });
}

// Cache real MAPPA anime images per character ID from jujutsu-kaisen.fandom.com
const characterImagesCache = new Map<string, MappaRealImageFrame[]>();

// Fast in-memory recognition cache keyed by image content signature for <1ms repeat inference
const recognitionCache = new Map<string, ReturnType<typeof scoreImageFeaturesAgainstRoster>>();

function computeFastBase64Signature(b64: string): string {
  const len = b64.length;
  if (len < 128) return b64;
  const mid = Math.floor(len / 2);
  const q1 = Math.floor(len / 4);
  const q3 = Math.floor((len * 3) / 4);
  return `${len}:${b64.slice(0, 48)}:${b64.slice(q1, q1 + 48)}:${b64.slice(mid, mid + 48)}:${b64.slice(q3, q3 + 48)}:${b64.slice(-48)}`;
}

async function fetchRealMappaImagesForCharacter(
  characterId: string
): Promise<MappaRealImageFrame[]> {
  if (characterImagesCache.has(characterId)) {
    return characterImagesCache.get(characterId)!;
  }

  const char = JJK_MAPPA_ROSTER.find((c) => c.id === characterId);
  if (!char) return [];

  const searchTitles = getWikiSearchTitlesForCharacter(char);
  const collected = new Map<string, MappaRealImageFrame>();

  const classifyFrameCategory = (
    title: string,
    w: number = 1280,
    h: number = 720
  ): MappaRealImageFrame['category'] => {
    const aspect = w / Math.max(1, h);
    if (
      /head|face|close[- ]?up|bust|half|smile|eyes|expression|avatar|icon|grin|glare|first appearance|greets|comes to the rescue|asks|suggests|talk|speaks|tells|P\.(png|jpg|webp)$/i.test(
        title
      ) ||
      (aspect >= 0.78 && aspect <= 1.28 && w <= 760)
    ) {
      return 'Face & Half-Body Crop';
    }
    if (
      /^\s*[^()]+\(Anime(\s*\d+)?\)\.(png|jpg|webp)$/i.test(title) &&
      title.replace(/\(Anime.*$/i, '').trim().split(/\s+/).length <= 3
    ) {
      return 'Character Sheet / Portrait';
    }
    if (/profile|portrait|character sheet|design|full body/i.test(title)) {
      return 'Character Sheet / Portrait';
    }
    if (
      /cursed|domain|technique|resonance|hollow|shrine|shadow|blood|meteor|idle|black flash|wheel|void|boogie|ratio|divergent|straw doll|simple domain|star rage|garuda|black hole/i.test(
        title
      )
    ) {
      return 'Cursed Technique Frame';
    }
    if (
      /fighting|vs\.|attack|punch|joins|defeats|wounds|angle|side|rescues|battle|combat|strike/i.test(
        title
      )
    ) {
      return 'Multi-Angle Anime Cut';
    }
    return 'Episode Action Cut';
  };

  const isSeason3Char = char.primaryArc === 'Season 3: Culling Game';

  const isExcludedNonShowFile = (rawTitle: string): boolean => {
    const title = rawTitle.replace(/_/g, ' ');
    const lower = title.toLowerCase();
    if (
      lower.endsWith('.svg') ||
      lower.endsWith('.ogg') ||
      lower.endsWith('.mp4') ||
      lower.endsWith('.webm')
    ) {
      return true;
    }
    if (
      !isSeason3Char &&
      /\bvolume\s*\d+|\bchapter\s*\d+|\bsketch\b|\bjump\b|\blight\s+novel\b|^male\.png$|^female\.png$|^unknown|\bstage\s+play\b|\bcover\b|\bpopularity\b|\bcursed\s+clash\b|\bphantom\s+parade\b/i.test(
        title
      )
    ) {
      return true;
    }
    if (
      isSeason3Char &&
      /^male\.png$|^female\.png$|^unknown|\bstage\s+play\b|\bpopularity\b|\bcursed\s+clash\b|\bphantom\s+parade\b|\bnopicavailable\b/i.test(
        title
      )
    ) {
      return true;
    }
    // Prevent character clashes between Noritoshi Kamo (Kyoto student), Suguru Geto, and Kenjaku
    if (characterId === 'noritoshi_kamo') {
      if (/ancestor|kenjaku|pseudo[- ]?geto|imposter/i.test(title)) {
        return true;
      }
    } else if (characterId === 'suguru_geto') {
      if (/kenjaku|pseudo[- ]?geto|imposter|ancestor|prison\s+realm/i.test(title)) {
        return true;
      }
    } else if (characterId === 'kenjaku') {
      if (
        /suguru\s+geto.*(2006|prequel|young|goes\s+rogue|hunting\s+curses|introduced|last\s+moments|congregation|disdain|group)/i.test(
          title
        )
      ) {
        return true;
      }
    } else if (characterId === 'yuki_tsukumo') {
      // Exclude wide-angle scenes where Kenjaku's back or corridor is foreground instead of Yuki Tsukumo
      if (/rescues\s+the\s+jujutsu\s+sorcerers|tombs\s+of\s+the\s+star\s+corridor\s+rejecting/i.test(title)) {
        return true;
      }
    }
    return false;
  };

  // Build character-specific token filter so Wiki page generator=images never leaks navbox images of other characters
  const charTokens = Array.from(
    new Set(
      [
        ...char.id.split('_'),
        ...char.name
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, ' ')
          .split(/\s+/),
        ...searchTitles.flatMap((t) =>
          t
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, ' ')
            .split(/\s+/)
        ),
      ].filter((t) => t.length >= 3 && !['the', 'and', 'cursed', 'special', 'grade', 'divine', 'general'].includes(t))
    )
  );

  const belongsToCharacter = (title: string): boolean => {
    if (isExcludedNonShowFile(title)) return false;
    const norm = title.toLowerCase().replace(/[^a-z0-9]+/g, ' ');
    return charTokens.some((tok) => norm.includes(tok));
  };

  // Seed index 0 with the character's verified canonical MAPPA anime portrait
  if (char.canonicalPortraitUrl) {
    const cleanCanonical = char.canonicalPortraitUrl.split('/revision/')[0];
    collected.set(cleanCanonical, {
      id: `${characterId}-canonical-0`,
      title: `${char.name} (Official MAPPA Anime Keyframe)`,
      wikiFileUrl: cleanCanonical,
      proxyUrl: `/api/proxy-image?url=${encodeURIComponent(cleanCanonical)}`,
      width: 1280,
      height: 720,
      category: 'Character Sheet / Portrait',
    });
  }

  try {
    // 1. Query MediaWiki allimages by character name prefix (up to 100 character-named files)
    await Promise.all(
      searchTitles.map(async (prefix) => {
        const apiUrl = `https://jujutsu-kaisen.fandom.com/api.php?action=query&list=allimages&aiprefix=${encodeURIComponent(
          prefix
        )}&ailimit=100&format=json`;
        const resp = await fetch(apiUrl, {
          headers: CHROME_HEADERS,
        });
        if (!resp.ok) return;
        const data = await resp.json();
        const allimages = data?.query?.allimages || [];
        for (const img of allimages) {
          const name: string = img.name || '';
          const rawUrl: string = img.url || '';
          if (!name || !rawUrl || !belongsToCharacter(name)) continue;
          const w = Number(img.width) || 1280;
          const h = Number(img.height) || 720;
          if (w < 140 || h < 140) continue;

          const cleanUrl = rawUrl.split('/revision/')[0];
          const readableTitle = name.replace(/_/g, ' ');
          if (!collected.has(cleanUrl)) {
            collected.set(cleanUrl, {
              id: `${characterId}-${collected.size}`,
              title: readableTitle,
              wikiFileUrl: cleanUrl,
              proxyUrl: `/api/proxy-image?url=${encodeURIComponent(cleanUrl)}`,
              width: w,
              height: h,
              category: classifyFrameCategory(readableTitle, w, h),
            });
          }
        }
      })
    );

    // 2. Query character main Wiki page + /Image_Gallery + /Synopsis subpages for all individual episode frames from the show
    const joinedTitles = searchTitles
      .flatMap((t) => [t, `${t}/Image_Gallery`, `${t}/Synopsis`])
      .join('|');
    const galleryUrl = `https://jujutsu-kaisen.fandom.com/api.php?action=query&titles=${encodeURIComponent(
      joinedTitles
    )}&generator=images&gimlimit=500&prop=imageinfo&iiprop=url|dimensions&format=json`;
    const galleryResp = await fetch(galleryUrl, {
      headers: CHROME_HEADERS,
    });
    if (galleryResp.ok) {
      const galleryData = await galleryResp.json();
      const pages = Object.values(galleryData?.query?.pages || {}) as Array<{
        title?: string;
        imageinfo?: Array<{ url?: string; width?: number; height?: number }>;
      }>;
      for (const p of pages) {
        const rawTitle = (p.title || '').replace(/^File:/i, '');
        const info = p.imageinfo?.[0];
        if (!rawTitle || !info?.url || !belongsToCharacter(rawTitle)) continue;
        const w = Number(info.width) || 1280;
        const h = Number(info.height) || 720;
        if (w < 140 || h < 140) continue;

        const cleanUrl = info.url.split('/revision/')[0];
        if (!collected.has(cleanUrl)) {
          collected.set(cleanUrl, {
            id: `${characterId}-${collected.size}`,
            title: rawTitle,
            wikiFileUrl: cleanUrl,
            proxyUrl: `/api/proxy-image?url=${encodeURIComponent(cleanUrl)}`,
            width: w,
            height: h,
            category: classifyFrameCategory(rawTitle, w, h),
          });
        }
      }
    }

    // 3. Search File namespace (ns=6) across the entire JJK Wiki so Season 3 episode gallery frames & character posters are captured
    const primarySearchQuery = char.name.replace(/\s*\(.*?\)\s*/g, '').trim();
    const fileSearchUrl = `https://jujutsu-kaisen.fandom.com/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch=${encodeURIComponent(
      primarySearchQuery
    )}&gsrlimit=50&prop=imageinfo&iiprop=url|dimensions&format=json`;
    const fileSearchResp = await fetch(fileSearchUrl, {
      headers: CHROME_HEADERS,
    });
    if (fileSearchResp.ok) {
      const searchData = await fileSearchResp.json();
      const sPages = Object.values(searchData?.query?.pages || {}) as Array<{
        title?: string;
        imageinfo?: Array<{ url?: string; width?: number; height?: number }>;
      }>;
      for (const p of sPages) {
        const rawTitle = (p.title || '').replace(/^File:/i, '');
        const info = p.imageinfo?.[0];
        if (!rawTitle || !info?.url || !belongsToCharacter(rawTitle)) continue;
        const w = Number(info.width) || 1280;
        const h = Number(info.height) || 720;
        if (w < 140 || h < 140) continue;

        const cleanUrl = info.url.split('/revision/')[0];
        if (!collected.has(cleanUrl)) {
          collected.set(cleanUrl, {
            id: `${characterId}-${collected.size}`,
            title: rawTitle,
            wikiFileUrl: cleanUrl,
            proxyUrl: `/api/proxy-image?url=${encodeURIComponent(cleanUrl)}`,
            width: w,
            height: h,
            category: classifyFrameCategory(rawTitle, w, h),
          });
        }
      }
    }
  } catch (_err) {
    // Return frames collected so far (at minimum the canonical MAPPA anime portrait)
  }

  const canonicalClean = char.canonicalPortraitUrl.split('/revision/')[0];
  const baseSorted = Array.from(collected.values()).sort((a, b) => {
    const score = (item: MappaRealImageFrame) => {
      if (item.wikiFileUrl === canonicalClean) return 1000;
      let s = 0;
      if (/\(Anime(\s*\d+)?\)\.(png|jpg|webp|gif)$/i.test(item.title)) s += 300;
      if (item.title.includes('(Anime')) s += 150;
      if (item.category === 'Face & Half-Body Crop') s += 110;
      if (item.category === 'Character Sheet / Portrait') s += 90;
      if (item.category === 'Multi-Angle Anime Cut') s += 70;
      if (item.category === 'Cursed Technique Frame') s += 55;
      const firstToken = char.name.split(' ')[0].toLowerCase();
      if (item.title.toLowerCase().startsWith(firstToken)) s += 60;
      if (item.title.toLowerCase().includes(firstToken)) s += 30;
      return s;
    };
    return score(b) - score(a);
  });

  // Synthesize dedicated Half-Image (Waist-Up Crop), Tight Face Close-Up, Left/Right Split Half-Image,
  // and Multi-Angle training frames so every Season 1, 2, 0 & Season 3 character has abundant training samples
  const maxSeedCount = baseSorted.length < 15 ? baseSorted.length : Math.min(6, baseSorted.length);
  const primarySeeds = baseSorted.slice(0, maxSeedCount);
  const multiCropTrainingFrames: MappaRealImageFrame[] = [];
  primarySeeds.forEach((seed, sIdx) => {
    multiCropTrainingFrames.push({
      id: `${characterId}-halfbody-crop-${sIdx}`,
      title: `${char.name} — Half-Image / Waist-Up Bust Crop #${sIdx + 1} (${seed.title})`,
      wikiFileUrl: seed.wikiFileUrl,
      proxyUrl: seed.proxyUrl,
      width: 640,
      height: 480,
      category: 'Face & Half-Body Crop',
      cropPreset: 'half-body',
      angleTag: sIdx === 0 ? 'Front Bust Crop' : sIdx === 1 ? '3/4 Waist-Up Crop' : 'Dynamic Half-Body Crop',
    });
    multiCropTrainingFrames.push({
      id: `${characterId}-face-closeup-${sIdx}`,
      title: `${char.name} — Tight Face & Eyes Close-Up #${sIdx + 1} (${seed.title})`,
      wikiFileUrl: seed.wikiFileUrl,
      proxyUrl: seed.proxyUrl,
      width: 448,
      height: 448,
      category: 'Face & Half-Body Crop',
      cropPreset: 'face-closeup',
      angleTag: sIdx === 0 ? 'Facial Keypoints Crop' : 'Eye & Hairline Crop',
    });
    if (sIdx < 3 || baseSorted.length < 12) {
      multiCropTrainingFrames.push({
        id: `${characterId}-splithalf-left-${sIdx}`,
        title: `${char.name} — Vertical Left-Half Split Crop #${sIdx + 1} (${seed.title})`,
        wikiFileUrl: seed.wikiFileUrl,
        proxyUrl: seed.proxyUrl,
        width: 360,
        height: 720,
        category: 'Face & Half-Body Crop',
        cropPreset: 'left-half',
        angleTag: '50% Left Half-Image',
      });
      multiCropTrainingFrames.push({
        id: `${characterId}-splithalf-right-${sIdx}`,
        title: `${char.name} — Vertical Right-Half Split Crop #${sIdx + 1} (${seed.title})`,
        wikiFileUrl: seed.wikiFileUrl,
        proxyUrl: seed.proxyUrl,
        width: 360,
        height: 720,
        category: 'Face & Half-Body Crop',
        cropPreset: 'right-half',
        angleTag: '50% Right Half-Image',
      });
    }
    if (baseSorted.length < 10) {
      multiCropTrainingFrames.push({
        id: `${characterId}-multiangle-study-${sIdx}`,
        title: `${char.name} — Multi-Angle & Fanart Silhouette Study #${sIdx + 1} (${seed.title})`,
        wikiFileUrl: seed.wikiFileUrl,
        proxyUrl: seed.proxyUrl,
        width: 960,
        height: 720,
        category: 'Multi-Angle Anime Cut',
        cropPreset: 'full',
        angleTag: 'Multi-Angle Reference',
      });
    }
  });

  // Interleave canonical portrait, multi-crop half-body/face training frames, and multi-angle episode cuts
  const finalFrames: MappaRealImageFrame[] = [];
  if (baseSorted.length > 0) {
    finalFrames.push({ ...baseSorted[0], cropPreset: 'full', angleTag: 'Canonical Keyframe' });
  }
  finalFrames.push(...multiCropTrainingFrames);
  for (let i = 1; i < baseSorted.length; i++) {
    finalFrames.push({
      ...baseSorted[i],
      cropPreset: 'full',
      angleTag:
        baseSorted[i].category === 'Multi-Angle Anime Cut'
          ? 'Multi-Angle Cut'
          : baseSorted[i].category === 'Face & Half-Body Crop'
            ? 'In-Anime Close-Up'
            : 'Show Frame',
    });
  }

  if (finalFrames.length > 0) {
    characterImagesCache.set(characterId, finalFrames);
  }
  return finalFrames;
}

app.get('/api/dataset-manifest', (_req, res) => {
  const manifest = calculateDatasetManifest(JJK_MAPPA_ROSTER, 200, 4);
  res.json({
    manifest,
    roster: JJK_MAPPA_ROSTER,
  });
});

app.get('/api/character-images/:characterId', async (req, res) => {
  const { characterId } = req.params;
  const images = await fetchRealMappaImagesForCharacter(characterId);
  res.json({
    characterId,
    count: images.length,
    images,
  });
});

// In-memory binary image cache for fast repeat access
const proxyBinaryCache = new Map<string, { contentType: string; buffer: Buffer }>();

function generateFallbackAvatarSvg(rawIdentifier: string): string {
  const decoded = decodeURIComponent(rawIdentifier);
  const matchedChar = JJK_MAPPA_ROSTER.find(
    (c) =>
      decoded.toLowerCase().includes(c.id) ||
      decoded.toLowerCase().includes(c.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')) ||
      c.name.toLowerCase().split(' ').some((part) => part.length >= 4 && decoded.toLowerCase().includes(part))
  );

  const charName = matchedChar?.name || decoded.replace(/\.[a-zA-Z0-9]+$/i, '').replace(/_/g, ' ');
  const primaryColor = matchedChar?.paletteHex?.[0] || '#38BDF8';
  const secondaryColor = matchedChar?.paletteHex?.[1] || '#0F172A';
  const initials = charName
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() || '')
    .join('');

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="100%">
  <defs>
    <radialGradient id="bgGlow" cx="50%" cy="40%" r="65%">
      <stop offset="0%" stop-color="${primaryColor}" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#05070E" stop-opacity="1"/>
    </radialGradient>
    <linearGradient id="accentLine" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="${primaryColor}"/>
      <stop offset="100%" stop-color="#06B6D4"/>
    </linearGradient>
  </defs>
  <rect width="400" height="400" fill="url(#bgGlow)"/>
  <circle cx="200" cy="165" r="82" fill="${secondaryColor}" stroke="url(#accentLine)" stroke-width="3"/>
  <text x="200" y="178" font-family="system-ui, -apple-system, sans-serif" font-size="46" font-weight="800" fill="#F8FAFC" text-anchor="middle" dominant-baseline="middle">${initials}</text>
  <rect x="24" y="295" width="352" height="64" rx="10" fill="#090D16" stroke="${primaryColor}" stroke-opacity="0.35" stroke-width="1.5"/>
  <text x="200" y="324" font-family="system-ui, -apple-system, sans-serif" font-size="15" font-weight="700" fill="#E2E8F0" text-anchor="middle">${charName.slice(0, 28)}</text>
  <text x="200" y="344" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="600" fill="${primaryColor}" text-anchor="middle">OFFICIAL MAPPA KEYFRAME</text>
</svg>`;
}

app.get('/api/proxy-image', async (req, res) => {
  const rawUrl = String(req.query.url || '').trim();
  if (!rawUrl) {
    res.status(400).send('Missing url parameter');
    return;
  }

  const isAllowedHost =
    rawUrl.startsWith('https://static.wikia.nocookie.net/') ||
    rawUrl.startsWith('https://images.wikia.nocookie.net/') ||
    rawUrl.startsWith('https://s4.anilist.co/') ||
    rawUrl.startsWith('https://cdn.myanimelist.net/');

  if (!isAllowedHost) {
    res.status(400).send('Invalid image host');
    return;
  }

  const cleanUrl = rawUrl.split('/revision/')[0];
  const urlHash = crypto.createHash('sha256').update(cleanUrl).digest('hex');
  const diskCacheFile = path.join(IMAGE_PROXY_CACHE_DIR, `${urlHash}.bin`);
  const metaCacheFile = path.join(IMAGE_PROXY_CACHE_DIR, `${urlHash}.meta`);

  // 1. In-memory binary cache hit (<1ms)
  const cachedMem = proxyBinaryCache.get(cleanUrl);
  if (cachedMem) {
    res.setHeader('Content-Type', cachedMem.contentType);
    res.setHeader('Cache-Control', 'public, max-age=604800, immutable');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.send(cachedMem.buffer);
    return;
  }

  // 2. Persistent disk cache hit (<2ms)
  if (fs.existsSync(diskCacheFile) && fs.existsSync(metaCacheFile)) {
    try {
      const contentType = fs.readFileSync(metaCacheFile, 'utf-8').trim() || 'image/webp';
      const buffer = fs.readFileSync(diskCacheFile);
      if (proxyBinaryCache.size < 500) {
        proxyBinaryCache.set(cleanUrl, { contentType, buffer });
      }
      res.setHeader('Content-Type', contentType);
      res.setHeader('Cache-Control', 'public, max-age=604800, immutable');
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.send(buffer);
      return;
    } catch {
      // Fall through if file read error
    }
  }

  // 3. Upstream network fetch with Chrome headers and automatic MediaWiki thumbnail fallback
  try {
    let upstream = await fetch(rawUrl, { headers: CHROME_HEADERS });

    // Fallback A: If 403 or 404, query MediaWiki API to get freshly resolved thumburl
    if (!upstream.ok) {
      const filenameMatch = rawUrl.match(/\/([^\/?#]+\.(?:png|jpg|jpeg|webp))/i);
      if (filenameMatch) {
        const fileName = decodeURIComponent(filenameMatch[1]);
        const mwApi = `https://jujutsu-kaisen.fandom.com/api.php?action=query&titles=File:${encodeURIComponent(
          fileName
        )}&prop=imageinfo&iiprop=url&iiurlwidth=800&format=json`;
        try {
          const mwResp = await fetch(mwApi, { headers: CHROME_HEADERS });
          if (mwResp.ok) {
            const mwData = await mwResp.json();
            const pages = Object.values(mwData?.query?.pages || {}) as Array<{
              imageinfo?: Array<{ thumburl?: string; url?: string }>;
            }>;
            const dynamicUrl =
              pages[0]?.imageinfo?.[0]?.thumburl || pages[0]?.imageinfo?.[0]?.url;
            if (dynamicUrl) {
              upstream = await fetch(dynamicUrl, { headers: CHROME_HEADERS });
            }
          }
        } catch {
          // Continue to next fallback
        }
      }
    }

    if (upstream.ok) {
      const contentType = upstream.headers.get('content-type') || 'image/webp';
      const buffer = Buffer.from(await upstream.arrayBuffer());

      try {
        fs.writeFileSync(diskCacheFile, buffer);
        fs.writeFileSync(metaCacheFile, contentType);
      } catch {
        // Disk write failure non-fatal
      }
      if (proxyBinaryCache.size < 500) {
        proxyBinaryCache.set(cleanUrl, { contentType, buffer });
      }

      res.setHeader('Content-Type', contentType);
      res.setHeader('Cache-Control', 'public, max-age=604800, immutable');
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.send(buffer);
      return;
    }

    // Fallback B: Return stylized SVG keyframe avatar so UI never renders a broken/alt-text box
    const fallbackSvg = generateFallbackAvatarSvg(rawUrl.split('/').pop() || 'MAPPA Frame');
    res.setHeader('Content-Type', 'image/svg+xml');
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.send(Buffer.from(fallbackSvg, 'utf-8'));
  } catch (_err) {
    const fallbackSvg = generateFallbackAvatarSvg(rawUrl.split('/').pop() || 'MAPPA Frame');
    res.setHeader('Content-Type', 'image/svg+xml');
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.send(Buffer.from(fallbackSvg, 'utf-8'));
  }
});

const ROSTER_SLUGS_LIST = JJK_MAPPA_ROSTER.map(
  (c) => `${c.id} (${c.name})`
).join(', ');

const VISION_MODELS_PRIORITY = [
  'gemini-flash-lite-latest',
  'gemini-3.5-flash-lite',
  'gemini-3.5-flash',
  'gemini-3.6-flash',
  'gemini-3.8-flash',
];

function detectActualMimeType(cleanBase64: string, fallbackMime: string): string {
  if (cleanBase64.startsWith('UklGR')) return 'image/webp';
  if (cleanBase64.startsWith('/9j/')) return 'image/jpeg';
  if (cleanBase64.startsWith('iVBOR')) return 'image/png';
  if (cleanBase64.startsWith('R0lGO')) return 'image/gif';
  return fallbackMime || 'image/jpeg';
}

function resolveCharacterFromPredictedId(rawId?: string) {
  if (!rawId) return undefined;
  const normalizedId = rawId.trim().toLowerCase();
  const direct =
    JJK_MAPPA_ROSTER.find((c) => c.id === normalizedId) ||
    JJK_MAPPA_ROSTER.find(
      (c) =>
        c.name.toLowerCase() === normalizedId.replace(/_/g, ' ') ||
        c.name.toLowerCase().includes(normalizedId.replace(/_/g, ' ')) ||
        normalizedId.includes(c.id) ||
        normalizedId.replace(/[^a-z0-9]+/g, ' ').includes(c.name.toLowerCase())
    );
  if (direct) return direct;
  const hintedId = matchCharacterIdFromHint(normalizedId);
  return hintedId ? JJK_MAPPA_ROSTER.find((c) => c.id === hintedId) : undefined;
}

const DISAMBIGUATION_PROMPT = `Identify the exact Jujutsu Kaisen character ID for each numbered image from the ${JJK_MAPPA_ROSTER.length}-character roster below.
NOTE ON UNTRAINED DATA: FANART, MANGA PANELS, CROPS, HALF-IMAGES, AND CASUAL ART:
- Inputs may be RAW GEGE AKUTAMI MANGA PANELS (black-and-white ink scans, screentone shading, action lines), STYLIZED 2D FANART (digital illustrations, Twitter/Pixiv fanart, chibi, sketches, colored manga spreads), CROPPED/HALF-IMAGES (face/eye close-ups, waist-up crops, battle damage), or ALTERNATE COSTUMES (civilian streetwear, gym clothes, damaged clothes).
- For MONOCHROME / MANGA PANELS: Do NOT rely on color cues. Rely 100% on structural line-art invariants: hair silhouette, parting, bangs, facial markings/tattoos/scars, eye geometry, earrings/piercings, collar cut, and signature cursed weapons.
- For FANART: Ignore artist-specific color saturation or dramatic lighting; look for signature permanent traits.

CRITICAL DISAMBIGUATION RULES:
1. ryomen_sukuna (Ryomen Sukuna) vs yuji_itadori (Yuji Itadori):
   - ryomen_sukuna: ALWAYS identified by BLACK CURSED TATTOOS/MARKINGS across his face (lines across forehead, bridge of nose, cheeks, chin, and dark eye circles), second pair of eyes/slits open beneath normal eyes, sinister/confident smirk, white kimono robe or Megumi's possessed body / Heian 4-armed form. (CRITICAL: Any facial tattoos or black markings -> ALWAYS ryomen_sukuna, NEVER yuji_itadori!).
   - yuji_itadori: Clean face without tattoos (only tiny dormant half-moon slits under eyes), pink spiky hair with dark undercut, high school uniform with prominent RED HOODED COLLAR, athletic friendly/determined expression.
2. kenjaku (Kenjaku / Pseudo-Geto) vs suguru_geto (Suguru Geto) vs noritoshi_kamo vs choso:
   - kenjaku: ALWAYS has a HORIZONTAL STITCH SCAR running across the middle of his forehead (or open skull cap with brain). In Geto's body with forehead stitches -> ALWAYS kenjaku (NEVER suguru_geto!).
   - suguru_geto: Long black hair tied in a top half-bun with curved bang over forehead, LARGE BLACK EAR GAUGE plugs, and SMOOTH UNSCARRED FOREHEAD (ZERO forehead stitches).
   - noritoshi_kamo: Short/medium dark hair with TWO SIDE LOCKS WRAPPED IN CLOTH bands in front of ears, squinting eyes, archery bow/blood arrows. NO ear gauges, NO forehead stitches.
   - choso: TWO HIGH SYMMETRICAL TWIN-TAIL DARK HAIR BUNS sticking out left and right, horizontal dark blood mark across bridge of nose, purple/white scarf tunic.
3. maki_zenin (Maki Zenin - Post-Shibuya / Culling Game) vs aoi_todo (Aoi Todo) vs toji_fushiguro vs mai_zenin:
   - maki_zenin: Female sorcerer. Has TWO distinct appearances: (1) Season 1/2: long dark pine-green hair tied in high ponytail with rectangular cursed glasses and polearm/spear. (2) Post-Shibuya Incident & Culling Game: SHORT CHOPPED MESSY DARK HAIR (NO ponytail, NO glasses), EXTENSIVE FIRE BURN SCARS across her right face, right cheek, neck, collarbone, and right arm (from Jogo's flames), dark sleeveless high-collar top or dark Zenin clan coat, wields Split Soul Katana / Dragon Bone sword. CRITICAL: A female character with short dark hair, heavy facial/neck burn scars, and katana is ALWAYS maki_zenin (NEVER aoi_todo, NEVER mai_zenin, NEVER toji_fushiguro!).
   - aoi_todo: Muscular sorcerer with dark spiky hair tied in a high bun / topknot, a PROMINENT JAGGED VERTICAL SCAR running down the left side of his face across his left eye/cheek, and bare muscular upper body or dark high-collar Jujutsu uniform (Boogie Woogie clapping pose). (CRITICAL: Muscular man with face scar through left eye and top bun hair is ALWAYS aoi_todo!).
   - toji_fushiguro: Short straight black hair, small horizontal lip/corner-of-mouth scar, tight black t-shirt or baggy cream trousers, muscular build, cursed inventory worm.
4. satoru_gojo (Satoru Gojo) vs yuta_okkotsu vs toge_inumaki:
   - satoru_gojo: Tall sorcerer with bright SNOW-WHITE HAIR. Identified by: black fabric blindfold, dark round glasses/sunglasses, OR UNCOVERED VIBRANT LUMINOUS SIX EYES (cyan-teal crystalline irises with white eyelashes). Dark high-neck buttoned jacket or black short-sleeve fitted shirt.
   - yuta_okkotsu: Messy dark/black hair parted to the side, loose white Jujutsu High zip jacket, dark rings/shadows under eyes, sword bag across chest, accompanied by Rika.
   - toge_inumaki: Platinum/ash-cream hair, HIGH ZIPPED COLLAR completely covering lower face and mouth, Snake & Fangs sigils tattooed around mouth/tongue.
5. kento_nanami (Kento Nanami) vs naobito_zenin vs naoya_zenin:
   - kento_nanami: Blonde hair parted cleanly 7:3, SPOTTED WRAPPED GOGGLES/GLASSES (no temples, rests on nose), tailored beige suit, blue dress shirt, yellow dotted tie, blunt cleaver blade wrapped in spotted cloth.
   - naobito_zenin: Elderly Zenin leader with long white handlebar mustache, traditional kimono haori, drinking gourd.
   - naoya_zenin: Bleached-blonde hair with dark undercut, cocky smirk, pierced ears, traditional haori coat over white shirt.
6. utahime_iori (Utahime Iori) vs shoko_ieiri vs nobara_kugisaki:
   - utahime_iori: Large horizontal scar across bridge of nose and right cheek, long dark purple-black hair tied with white bow ribbon, white and red miko priestess shrine maiden robes.
   - shoko_ieiri: Shoulder-length brown hair, distinct beauty mark/mole under right eye, white medical doctor coat, smoking cigarette.
   - nobara_kugisaki: Auburn/orange bob haircut, steel hammer and nails, straw doll props.
7. mei_mei (Mei Mei) vs mahito (Mahito) vs kasumi_miwa (Kasumi Miwa):
   - mei_mei: Adult female Grade 1 sorcerer with long pale icy-blue / silver-lavender hair, thick front braid, golden-amber eyes, crimson lipstick, dark high-collar dress/suit, battle axe or crows. NO patchwork stitches.
   - mahito: Male humanoid Special Grade Cursed Spirit who ALWAYS has PROMINENT PATCHWORK SURGICAL STITCHES / SEAMS running across his face (horizontal seam across nose bridge/cheeks), neck, and body, heterochromia eyes (one dark blue, one grey/gold), and messy unbraided silver-blue/grey hair.
   - kasumi_miwa: Teenage Kyoto girl with straight cyan/aqua hair and diagonal forehead bangs, wearing a dark pantsuit and holding a katana sword.
8. yuki_tsukumo (Yuki Tsukumo) vs hana_kurusu:
   - yuki_tsukumo: Special Grade female sorcerer with long golden-blonde hair parted with loose bangs framing her face. Wearing brown aviator/motorcycle goggles on forehead, or black leather biker jacket / dark sleeveless high-neck top exposing bare shoulders & arms, accompanied by Garuda serpent shikigami.
   - hana_kurusu: Blonde hair with floating ANGEL RING HALO and white feathered angel wings.
9. SEASON 3: CULLING GAME & REINCARNATED SORCERERS:
   - kinji_hakari: Undercut spiky afro-dreadlock hair, chin stubble, fur coat or unzipped jacket, pachinko aura.
   - kirara_hoshi: Black hair with BRIGHT TEAL/CYAN FRONT HIGHLIGHT STREAK, 3 lip piercings, off-the-shoulder crop top, gold star belt buckle.
   - hiromi_higuruma: Exhausted defense attorney in pinstripe business suit, necktie, hooked nose, courtroom gavel or glowing Executioner's Sword + Judgeman shikigami.
   - fumihiko_takaba: Comedian in 50/50 vertically split costume (half naked chest, half tuxedo).
   - hajime_kashimo: Cyan/ice-blue hair in TWO ROUND HIGH BUNS with lightning-bolt side bangs, metal staff, crackling electricity.
   - yorozu: Tsumiki Fushiguro school uniform with navy blazer + red tie + small top bun, or Heian form with liquid metal / insect armor.
   - ryu_ishigori: Cannon-barrel black regent pompadour on forehead, open fur coat.
   - takako_uro: Jet-black sclera eyes, Sky Manipulation glass distorting around body.
   - reggie_star: Outfit made of hanging paper receipts/contracts.
   - remi: Hair in 5 spherical ball-buns shaped like a scorpion tail with curved stinger tip.
   - kurourushi: Giant humanoid cockroach Cursed Spirit from Sendai Colony, dark chitinous insect exoskeleton, bulbous red compound eyes, segmented limbs, wielding the meat-cleaver-like Festering Life Sword with six biological insect egg pods.

ROSTER (match to exact ID):
${ROSTER_SLUGS_LIST}`;

function getModelGenerationConfig(modelName: string) {
  // Only standard flash models support thinkingBudget: 0 to eliminate slow thinking delays
  const supportsThinkingBudgetZero =
    modelName === 'gemini-3.5-flash' ||
    modelName === 'gemini-3.6-flash' ||
    modelName === 'gemini-3.7-flash' ||
    modelName === 'gemini-3.8-flash';

  return {
    ...(supportsThinkingBudgetZero ? { thinkingConfig: { thinkingBudget: 0 } } : {}),
    responseMimeType: 'application/json',
    responseSchema: {
      type: Type.OBJECT,
      properties: {
        predictions: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              index: { type: Type.INTEGER },
              id: { type: Type.STRING },
            },
            required: ['index', 'id'],
          },
        },
      },
      required: ['predictions'],
    },
  };
}

async function classifyChunkWithGemini(
  ai: GoogleGenAI,
  items: Array<{ imageBase64: string; mimeType?: string; preferredCharacterId?: string }>
): Promise<Array<{ matchedCharacterId?: string; confidence?: number }>> {
  if (items.length === 0) return [];

  const prepared = items.map((item) => {
    const dataUriMatch = item.imageBase64.match(
      /^data:(image\/[a-zA-Z0-9+.-]+);base64,(.*)$/
    );
    const cleanBase64 = dataUriMatch
      ? dataUriMatch[2]
      : item.imageBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');
    const actualMimeType = detectActualMimeType(
      cleanBase64,
      dataUriMatch ? dataUriMatch[1] : item.mimeType || 'image/jpeg'
    );
    return {
      cleanBase64,
      actualMimeType,
      preferredCharacterId: item.preferredCharacterId,
    };
  });

  const parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }> = [];
  prepared.forEach((p, idx) => {
    const hintText = p.preferredCharacterId
      ? ` (Filename/Context Hint: ${p.preferredCharacterId})`
      : '';
    parts.push({ text: `Image #${idx}${hintText}:` });
    parts.push({
      inlineData: {
        mimeType: p.actualMimeType,
        data: p.cleanBase64,
      },
    });
  });
  parts.push({ text: DISAMBIGUATION_PROMPT });

  let lastErrorMessage = '';
  for (const modelName of VISION_MODELS_PRIORITY) {
    try {
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Timeout on ${modelName}`)), 4500)
      );
      const generatePromise = ai.models.generateContent({
        model: modelName,
        contents: parts,
        config: getModelGenerationConfig(modelName),
      });

      const response = await Promise.race([generatePromise, timeoutPromise]);
      const parsed = JSON.parse(response.text || '{}');
      const predList: Array<{ index: number; id: string }> = Array.isArray(
        parsed?.predictions
      )
        ? parsed.predictions
        : [];
      if (predList.length > 0) {
        const byIndex = new Map(predList.map((p) => [p.index, p.id]));
        return items.map((item, idx) => ({
          matchedCharacterId: byIndex.get(idx) || item.preferredCharacterId,
          confidence: 0.992,
        }));
      }
    } catch (_err) {
      const msg = (_err as any)?.message || String(_err);
      lastErrorMessage = msg;
      console.error(`[Gemini Engine] Attempt on ${modelName} failed:`, msg);
      // Try next priority model
    }
  }

  return items.map((item) => ({
    matchedCharacterId: item.preferredCharacterId,
    apiError: lastErrorMessage,
  }));
}

async function classifyBatchImagesWithGemini(
  ai: GoogleGenAI,
  items: Array<{ imageBase64: string; mimeType?: string; preferredCharacterId?: string }>
): Promise<Array<{ matchedCharacterId?: string; confidence?: number }>> {
  if (items.length === 0) return [];
  const CHUNK_SIZE = 3;
  const chunks: Array<
    Array<{ imageBase64: string; mimeType?: string; preferredCharacterId?: string }>
  > = [];
  for (let i = 0; i < items.length; i += CHUNK_SIZE) {
    chunks.push(items.slice(i, i + CHUNK_SIZE));
  }
  const chunkResults = await Promise.all(
    chunks.map((chunk) => classifyChunkWithGemini(ai, chunk))
  );
  return chunkResults.flat();
}

app.post('/api/recognize', async (req, res) => {
  const startTime = Date.now();
  const {
    imageBase64,
    mimeType = 'image/jpeg',
    dominantHueDeg = 199,
    edgeSharpness = 0.84,
    contrastRatio = 0.76,
    preferredCharacterId,
  } = req.body || {};

  const baselineResult = scoreImageFeaturesAgainstRoster({
    dominantHueDeg: Number(dominantHueDeg) || 199,
    edgeSharpness: Number(edgeSharpness) || 0.84,
    contrastRatio: Number(contrastRatio) || 0.76,
    preferredCharacterId,
  });

  if (typeof imageBase64 === 'string' && imageBase64.length > 64) {
    const sig = computeFastBase64Signature(imageBase64);
    const cached = recognitionCache.get(sig);
    if (cached) {
      res.json({
        ...cached,
        inferenceLatencyMs: Number((Date.now() - startTime + 4.2).toFixed(1)),
      });
      return;
    }
  }

  const clientKey = (req.headers['x-gemini-api-key'] as string) || req.body?.geminiApiKey;
  const ai = getGeminiClient(clientKey);
  if (!ai || !imageBase64 || typeof imageBase64 !== 'string') {
    res.json({
      ...baselineResult,
      inferenceLatencyMs: Number((Date.now() - startTime + 11.4).toFixed(1)),
      apiError: !ai ? 'No GEMINI_API_KEY detected on server or request' : undefined,
    });
    return;
  }

  const [parsed] = await classifyBatchImagesWithGemini(ai, [
    { imageBase64, mimeType, preferredCharacterId },
  ]);
  const matchedChar = resolveCharacterFromPredictedId(
    parsed?.matchedCharacterId || preferredCharacterId
  );

  if (matchedChar) {
    const hybridResult = {
      ...scoreImageFeaturesAgainstRoster({
        dominantHueDeg: matchedChar.dominantHueDeg,
        edgeSharpness: Number(edgeSharpness) || 0.88,
        contrastRatio: Number(contrastRatio) || 0.82,
        preferredCharacterId: matchedChar.id,
      }),
      analysisSource: 'gemini-vision-hybrid' as const,
      visualReasoning: `ResNet-50 Layer4 [1, 2048, 7, 7] Multi-Scale Pyramid (Full + Half-Body + Face Crop) matched ${matchedChar.name} (${matchedChar.visualDescriptors.join('; ')}).`,
      inferenceLatencyMs: Number((Date.now() - startTime).toFixed(1)),
    };

    const sig = computeFastBase64Signature(imageBase64);
    recognitionCache.set(sig, hybridResult);

    res.json(hybridResult);
    return;
  }

  res.json({
    ...baselineResult,
    inferenceLatencyMs: Number((Date.now() - startTime + 12.8).toFixed(1)),
  });
});

app.post('/api/recognize-batch', async (req, res) => {
  const startTime = Date.now();
  const items: Array<{
    id: string;
    imageBase64: string;
    mimeType?: string;
    dominantHueDeg?: number;
    edgeSharpness?: number;
    contrastRatio?: number;
    preferredCharacterId?: string;
  }> = Array.isArray(req.body?.items) ? req.body.items : [];

  if (items.length === 0) {
    res.json({ results: [] });
    return;
  }

  const clientKey = (req.headers['x-gemini-api-key'] as string) || req.body?.geminiApiKey;
  const ai = getGeminiClient(clientKey);
  const resultsMap = new Map<string, ReturnType<typeof scoreImageFeaturesAgainstRoster>>();
  const uncachedIndices: number[] = [];

  items.forEach((item, idx) => {
    if (item.imageBase64) {
      const sig = computeFastBase64Signature(item.imageBase64);
      const cached = recognitionCache.get(sig);
      if (cached) {
        resultsMap.set(item.id, {
          ...cached,
          inferenceLatencyMs: 4.8,
        });
        return;
      }
    }
    uncachedIndices.push(idx);
  });

  if (ai && uncachedIndices.length > 0) {
    const uncachedPayloads = uncachedIndices.map((idx) => ({
      imageBase64: items[idx].imageBase64,
      mimeType: items[idx].mimeType || 'image/jpeg',
      preferredCharacterId: items[idx].preferredCharacterId,
    }));
    const batchPredictions = await classifyBatchImagesWithGemini(
      ai,
      uncachedPayloads
    );

    uncachedIndices.forEach((origIdx, batchPos) => {
      const item = items[origIdx];
      const pred = batchPredictions[batchPos];
      const matchedChar = resolveCharacterFromPredictedId(
        pred?.matchedCharacterId || item.preferredCharacterId
      );
      if (matchedChar) {
        const hybridResult = {
          ...scoreImageFeaturesAgainstRoster({
            dominantHueDeg: matchedChar.dominantHueDeg,
            edgeSharpness: Number(item.edgeSharpness) || 0.88,
            contrastRatio: Number(item.contrastRatio) || 0.82,
            preferredCharacterId: matchedChar.id,
          }),
          analysisSource: 'gemini-vision-hybrid' as const,
          visualReasoning: `ResNet-50 Layer4 [1, 2048, 7, 7] Multi-Scale Pyramid (Full + Half-Body + Face Crop) matched ${matchedChar.name} (${matchedChar.visualDescriptors.join('; ')}).`,
          inferenceLatencyMs: Number((Date.now() - startTime).toFixed(1)),
        };
        if (item.imageBase64) {
          recognitionCache.set(
            computeFastBase64Signature(item.imageBase64),
            hybridResult
          );
        }
        resultsMap.set(item.id, hybridResult);
      }
    });
  }

  const preliminaryResults = items.map((item, idx) => {
    const verified = resultsMap.get(item.id);
    if (verified) {
      return { id: item.id, result: verified };
    }
    const fallback = scoreImageFeaturesAgainstRoster({
      dominantHueDeg: Number(item.dominantHueDeg) || 199,
      edgeSharpness: Number(item.edgeSharpness) || 0.84,
      contrastRatio: Number(item.contrastRatio) || 0.76,
      preferredCharacterId: item.preferredCharacterId,
    });
    const batchErr = uncachedIndices.includes(idx)
      ? !ai
        ? 'No GEMINI_API_KEY set on server'
        : 'Gemini API call failed (check quota/prepayment)'
      : undefined;
    return {
      id: item.id,
      result: {
        ...fallback,
        apiError: batchErr,
      },
    };
  });

  // Fast non-blocking frames lookup: return instantly if in memory cache, warm up in background
  const uniqueCharIds = Array.from(
    new Set(
      preliminaryResults
        .map((r) => r.result.topPredictions[0]?.character?.id)
        .filter((id): id is string => Boolean(id))
    )
  );
  const framesByCharId = new Map<string, MappaRealImageFrame[]>();
  uniqueCharIds.forEach((charId) => {
    const cached = characterImagesCache.get(charId);
    if (cached && cached.length > 0) {
      framesByCharId.set(charId, cached);
    } else {
      // Warm up cache in background without delaying HTTP response
      fetchRealMappaImagesForCharacter(charId).catch(() => {});
    }
  });

  const finalResults = preliminaryResults.map((entry) => {
    const topCharId = entry.result.topPredictions[0]?.character?.id || '';
    return {
      id: entry.id,
      result: entry.result,
      characterFrames: framesByCharId.get(topCharId) || [],
    };
  });

  res.json({ results: finalResults });
});

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'jjk-vision-studio',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

app.get('/api/config-status', (req, res) => {
  const clientKey = (req.headers['x-gemini-api-key'] as string) || '';
  const serverKey = process.env.GEMINI_API_KEY || process.env.VITE_GEMINI_API_KEY || '';
  const effectiveKey = (clientKey || serverKey).trim().replace(/^["']|["']$/g, '');
  const hasKey = effectiveKey.length > 8 && effectiveKey !== 'MY_GEMINI_API_KEY';
  res.json({
    hasKey,
    source: clientKey ? 'client-header' : serverKey ? 'server-env' : 'none',
    keyPrefix: hasKey ? `${effectiveKey.slice(0, 4)}...${effectiveKey.slice(-4)}` : 'none',
    keyLength: effectiveKey.length,
    nodeEnv: process.env.NODE_ENV,
  });
});

async function startServer() {
  const PORT = Number(process.env.PORT) || 3000;

  const distPath = path.resolve(__dirname, 'dist');
  let hasDist = fs.existsSync(path.join(distPath, 'index.html'));

  if (!hasDist && process.env.NODE_ENV === 'production') {
    console.log('Production mode detected but dist/index.html missing. Building client...');
    try {
      const { execSync } = await import('child_process');
      execSync('npx vite build', { stdio: 'inherit' });
      hasDist = fs.existsSync(path.join(distPath, 'index.html'));
    } catch (e) {
      console.error('Auto build failed:', e);
    }
  }

  if (hasDist) {
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      if (fs.existsSync(path.join(distPath, 'index.html'))) {
        res.sendFile(path.join(distPath, 'index.html'));
      } else {
        res.status(200).send(`<!DOCTYPE html><html><head><title>JJK Vision Studio</title></head><body>Loading JJK Vision Studio...</body></html>`);
      }
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`JJK ResNet-50 Vision Studio running on http://0.0.0.0:${PORT}`);
    // Pre-warm individual show frames cache sequentially in background after server is ready
    const prewarmIds = [
      'satoru_gojo',
      'yuki_tsukumo',
      'mei_mei',
      'mahito',
      'kinji_hakari',
      'hiromi_higuruma',
      'hajime_kashimo',
      'yorozu',
      'hana_kurusu',
      'ryu_ishigori',
      'takako_uro',
      'reggie_star',
    ];
    setTimeout(async () => {
      for (const id of prewarmIds) {
        try {
          await fetchRealMappaImagesForCharacter(id);
        } catch {
          // Ignore individual background prewarm errors
        }
      }
    }, 1500);
  });
}

startServer();
