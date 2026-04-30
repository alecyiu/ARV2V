"""Themed prompt presets for the grid demo (4 themes; matches 4 non-GT columns).

Prompts are English narrative-style descriptions of a forward-facing dashcam
view, listing concrete visible elements (snowflakes, tire tracks, neon signs,
cacti, etc.) so the model has plenty of cues to lock onto. Negatives are
comma-separated lists covering both generic Wan failure modes (color cast,
blur, deformed cars, watermarks) and theme-specific failures we have observed
in earlier runs (e.g. blue cast on snow, whole-wall glow on cyberpunk,
yellow filter on desert).

Each preset carries its own `recommended_conditioning_scale` because the
optimal source-vs-prompt balance depends on how transformative the edit is:
  - low (0.4): scene-replacement edits (e.g. buildings -> desert dunes) need
    the source signal weak so the prompt can put new content in.
  - medium (0.5-0.6): texture/atmosphere edits (snow accumulation, neon
    signage on walls) need balance.
  - high (0.75): style/render-only edits (cartoon redraw of the same scene)
    need the source signal strong so composition is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPreset:
    name: str
    label: str
    prompt: str
    negative_prompt: str
    recommended_conditioning_scale: float = 0.5


SNOW = PromptPreset(
    name="snow",
    label="Snow",
    recommended_conditioning_scale=0.5,
    prompt=(
        "The video depicts a forward-facing dashcam view of a snowy winter "
        "street. Heavy snowflakes fall through the air, drifting diagonally "
        "across the camera as the vehicle drives forward. The road is covered "
        "in a thick layer of fresh white snow with parallel tire tracks "
        "carved through it. Parked and moving cars are dusted with snow on "
        "their roofs, hoods, and windshields. Sidewalks, lawns, rooftops, "
        "and parked vehicles are blanketed in pristine white. Bare tree "
        "branches sag under the weight of snow. Streetlamps, fire hydrants, "
        "mailboxes, and traffic signs wear thick caps of accumulated snow. "
        "The sky is overcast and pale gray, casting cold diffuse winter "
        "light with no harsh shadows. Cool blue-white tones dominate; the "
        "air is misty with light snowfall. Distant buildings fade into a "
        "soft white haze. The road glistens faintly where light catches "
        "packed snow and ice. Passing cars kick up brief clouds of fine "
        "powdered snow behind them. Pedestrians in heavy coats walk briskly "
        "with scarves pulled up. Photorealistic, cinematic dashcam footage, "
        "real-world winter weather, natural overcast lighting."
    ),
    negative_prompt=(
        "yellow color cast, orange tint, warm tones, overexposed, "
        "oversaturated colors, washed out, overall gray, worst quality, "
        "low quality, JPEG compression residue, blurred details, motion "
        "blur smear, ghosting, static frame, frozen motion, still picture, "
        "cartoon, anime, painting, illustration, sketch, CGI render, video "
        "game graphics, 3D render, surreal, distorted geometry, warped "
        "buildings, melting cars, deformed vehicles, asymmetric wheels, "
        "missing windows, floating cars, duplicated vehicles, unrealistic "
        "shadows, fisheye distortion, lens flare, dirty lens, water "
        "droplets on lens, watermark, logo, subtitles, text overlay, "
        "timestamp, tilted horizon, wrong perspective, mirrored image, "
        "upside down, neon lights, neon signs, cyberpunk, futuristic "
        "megacity, holographic billboards, glowing magenta cyan pink, dark "
        "night street, flying cars, blade runner, sand dunes, desert, "
        "cacti, saguaro, palm trees, sun-bleached arid pavement, heat "
        "shimmer, sandy ground, dust storm, sunny clear blue sky, harsh "
        "midday sun, summer foliage, rainforest, lush green vegetation, "
        "dry pavement, no snow on ground, no snowfall, melted snow, "
        "summer scene"
    ),
)


MARIO_KART = PromptPreset(
    name="mario_kart",
    label="Mario Kart",
    recommended_conditioning_scale=0.75,
    prompt=(
        "The video depicts a forward-facing dashcam view of a street "
        "redrawn in the visual style of Nintendo's Mario Kart cartoon "
        "racing game, while completely preserving the original scene's "
        "composition, camera angle, object positions, and silhouettes. "
        "Every surface is rendered with thick uniform black outlines, "
        "cel-shaded flat fills, and a saturated candy-bright palette of "
        "primary reds, blues, yellows, greens, and cartoon white. The same "
        "vehicles in the same positions are redrawn as cartoon karts with "
        "chunky exaggerated wheels, rounded simplified shapes, and bright "
        "playful liveries. Buildings keep their original silhouettes but "
        "are drawn as flat bright color blocks with cute cartoon windows, "
        "doors, and rooftops. The road remains the same street but is "
        "rendered as cartoon asphalt with bold colored lane markings and "
        "large arcade-style arrow decals. The sky is a cheerful cartoon "
        "blue dotted with fluffy stylized white clouds and a smiling "
        "golden cartoon sun. Lighting is high-key, flat, and playful with "
        "minimal shading. Trees, lampposts, and street furniture are "
        "stylized into rounded chunky cartoon versions of themselves. "
        "Whimsical decorative elements appear along the shoulder: oversize "
        "traffic cones, glowing question-mark item boxes, banana peel "
        "pickups, and stylized palm trees. The result is a stylized "
        "cartoon redraw of the same dashcam scene — composition, camera "
        "angle, vehicle placement, and recognizability of every object "
        "are preserved exactly, only the rendering style changes."
    ),
    negative_prompt=(
        "photorealistic, photoreal, realistic photograph, dashcam footage "
        "as-is, documentary style, real-world physics, subtle natural "
        "lighting, muted natural colors, naturalistic shadows, scene "
        "completely replaced, original composition destroyed, background "
        "turned solid black, source content removed, hallucinated "
        "unrelated scene, scene wiped, snow, snowfall, winter, blizzard, "
        "frost, ice, overcast gray sky, blanket of white snow, slush, "
        "neon lights, neon signs, cyberpunk, futuristic megacity, "
        "holographic billboards, glowing magenta cyan pink, wet "
        "rain-slicked neon-lit asphalt, dark night street, flying cars, "
        "blade runner, sci-fi, desert, sand dunes, sandy ground, cacti, "
        "saguaro, palm trees, arid biome, sun-bleached pavement, heat "
        "shimmer, dust storm, rocky outcrops, mesa, southwestern landscape, "
        "motion blur smear, ghosting, static frame, frozen motion, still "
        "picture, deformed vehicles, melting cars, asymmetric wheels, "
        "missing windows, floating cars, duplicated vehicles, warped "
        "buildings, fisheye distortion, lens flare, dirty lens, water "
        "droplets on lens, washed out colors, dull desaturated tones, "
        "gray gloom, watermark, logo, subtitles, text overlay, timestamp, "
        "tilted horizon, wrong perspective, mirrored image, upside down, "
        "worst quality, low quality, JPEG compression residue, blurred "
        "details"
    ),
)


CYBERPUNK = PromptPreset(
    name="cyberpunk",
    label="Cyberpunk",
    recommended_conditioning_scale=0.6,
    prompt=(
        "The video depicts a forward-facing dashcam view of a street "
        "through a neon-lit cyberpunk megacity at deep night, in the "
        "cinematic style of Blade Runner and Ghost in the Shell. The "
        "original buildings, road, sidewalks, and vehicles all keep their "
        "structure and silhouettes clearly visible — the architecture is "
        "intact, walls remain dark, only their surfaces are decorated with "
        "concrete glowing signage. Discrete neon signs hang near windows "
        "and storefronts as individual luminous objects mounted on "
        "otherwise dark walls: glowing Chinese and Japanese kanji shop "
        "placards, animated LED billboards, flickering neon-tube logos in "
        "pink, cyan, magenta, and electric blue, lit-up signboard panels, "
        "and corporate logo lightboxes. Towering futuristic skyscrapers "
        "loom in the smoggy distance, dotted with scattered points of "
        "neon light. The wet rain-slicked asphalt reflects the signs in "
        "smeared colored puddles, while volumetric steam columns rise "
        "from manhole grates and ventilation shafts. The sky is dark and "
        "hazy with light-pollution glow above the skyline. Vehicles "
        "retain their original shapes, augmented with sleek LED running "
        "lights, headlights, and tail lights. Silhouettes of pedestrians "
        "in long coats are backlit by saturated artificial light. The "
        "atmosphere is dystopian, atmospheric, and cinematic with strong "
        "contrast — bright neon highlights only on the signage, "
        "reflections, and vehicle lights, surrounded by deep shadow on "
        "building walls and street pavement. Photorealistic, dark "
        "cinematic color grade, high-contrast night photography."
    ),
    negative_prompt=(
        "entire wall glowing solid color, walls turned into pure colored "
        "light, neon flooding the whole image, building structure "
        "dissolved by light wash, full-screen monochrome glow, walls "
        "rendered as red-only or green-only or blue-only, no discrete "
        "signs visible, signs missing, color cast over everything, snow, "
        "snowfall, winter, blizzard, frost, ice, overcast daytime gray "
        "sky, blanket of white snow, slush, daytime, bright daylight, "
        "sunny clear sky, harsh noon sun, morning fog, sunset, cartoon, "
        "anime, cel-shaded, thick black outlines, mario kart, kart "
        "racing, video game graphics, item boxes, checkered race stripes, "
        "oversize cartoon karts, low-poly, desert, sand dunes, sandy "
        "ground, cacti, saguaro, palm trees, arid biome, sun-bleached "
        "pavement, heat shimmer, dust storm, rocky outcrops, mesa, "
        "southwestern landscape, dry suburban street, rural countryside, "
        "lush green forest, washed out colors, dull desaturated tones, "
        "motion blur smear, ghosting, static frame, frozen motion, still "
        "picture, deformed vehicles, melting cars, asymmetric wheels, "
        "missing windows, floating cars, duplicated vehicles, warped "
        "buildings, fisheye distortion, lens flare, dirty lens, water "
        "droplets on lens, watermark, logo, subtitles, text overlay, "
        "timestamp, tilted horizon, wrong perspective, mirrored image, "
        "upside down, worst quality, low quality, JPEG compression "
        "residue, blurred details"
    ),
)


DESERT = PromptPreset(
    name="desert",
    label="Desert",
    recommended_conditioning_scale=0.4,
    prompt=(
        "The video depicts a forward-facing dashcam view of a road that "
        "has been completely transformed into a sun-baked highway running "
        "through a vast arid southwestern American desert biome. The "
        "original buildings, houses, and storefronts that once lined the "
        "street are gone — replaced entirely with rolling golden sand "
        "dunes, red rocky outcrops, weathered boulders, and distant "
        "flat-topped mesas stretching to a hazy horizon. Tall saguaro "
        "cacti, barrel cacti, mesquite shrubs, sagebrush, and dry brush "
        "line the shoulders of the road, with the occasional tumbleweed "
        "rolling lazily across the lanes. The asphalt itself is "
        "sun-bleached pale gray, cracked, and partially buried beneath "
        "thin drifts of fine windblown sand. The sky is bright, "
        "cloudless, and deep blue with the harsh midday sun directly "
        "overhead, casting sharp dark crisp shadows beneath every object. "
        "Visible heat shimmer and mirage waves rise from the hot "
        "pavement, warm dust hangs in the air, and distant dust devils "
        "swirl lazily across the dunes. Weathered wooden signposts, "
        "rusted guardrails, and faded mile markers stand alongside the "
        "road. Any vehicles on the road are dust-covered pickup trucks "
        "and battered cars typical of a Mojave-like landscape. Heat haze "
        "blurs the horizon. Photorealistic, cinematic, harsh natural "
        "high-noon sunlight, warm golden tones, dry arid atmosphere."
    ),
    negative_prompt=(
        "yellow color cast over original scene, orange filter applied to "
        "existing footage, just a color tint, original buildings still "
        "visible, original houses still standing, urban suburb unchanged, "
        "telephone poles in residential street, residential street "
        "remains, original storefronts intact, source scene unchanged, "
        "only color graded, snow, snowfall, winter, blizzard, frost, ice, "
        "overcast snowy gray sky, blanket of white snow, neon lights, "
        "neon signs, holographic billboards, cyberpunk, futuristic "
        "megacity, magenta cyan pink glow, wet rain-slicked neon street, "
        "flying cars, sci-fi, blade runner, dark night street, cartoon, "
        "anime, cel-shaded, thick black outlines, mario kart, kart "
        "racing, video game graphics, item boxes, checkered race stripes, "
        "oversize cartoon karts, low-poly, lush green forest, rainforest, "
        "dense vegetation, urban skyscrapers, dense city traffic, "
        "suburban tree-lined street, rain, fog, overcast clouds, "
        "snow-covered terrain, washed out colors, dull desaturated tones, "
        "motion blur smear, ghosting, static frame, frozen motion, still "
        "picture, deformed vehicles, melting cars, asymmetric wheels, "
        "missing windows, floating cars, duplicated vehicles, warped "
        "buildings, fisheye distortion, lens flare, dirty lens, water "
        "droplets on lens, watermark, logo, subtitles, text overlay, "
        "timestamp, tilted horizon, wrong perspective, mirrored image, "
        "upside down, worst quality, low quality, JPEG compression "
        "residue, blurred details"
    ),
)


PRESETS: dict[str, PromptPreset] = {
    p.name: p for p in (SNOW, MARIO_KART, CYBERPUNK, DESERT)
}
