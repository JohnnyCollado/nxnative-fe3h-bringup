# Fire Emblem Fortune's Weave (`04007EF01F8AE000`)

A Switch 2 game: v131072, SDK 22.2.0 built for the Switch 2 (`Ounce`), NVN2 API 231.40. Intelligent Systems' engine on NintendoWare (G3d, Ui2d, Vfx, Atk), PhysX 5.6, a deferred renderer. Modules `main`, `subsdk2` (movies) and `sdk`; `subsdk0` (DLSS) and `subsdk1` (CUDA) are not loaded. **Plays on the Mac**: the opening movie, the dialogue and the first battle map, whose first turn matches published footage, plus battle scenes in gameplay testing.

## What it needs

| Setting | Where | Why |
|---|---|---|
| `mem_alias 0x400000000`, `mem_reserve 0x550000000`, `mem_stack 0x10000000`, `app_pool 0x240000000` | [`game.tsv`](game.tsv) | Splatoon Raiders' split as a starting point; the Switch 2's 9 GB |
| `sys_resource 0x4000000` | `game.tsv` | the NPDM declares 64 MB, so virtual address memory is on |
| `pad_layout full` | `game.tsv` | the game asks for most of the controller at once |
| `zip_store .zst:.mp4:.webm:.metallib` | `game.tsv` | the RomFS's files are compressed already |

No `render_cond`: not tried, and nothing has called for it so far.

## Files

- A dump folder holding only `exefs/` and `romfs/`, with no `base/` layout: a RomFS of 28 GB. The NSOs are ZBIC-compressed, so tools that read them need the `zbic` package in the work folder's venv.
- Shaders: 16,888 in [`shader_map.tsv`](shader_map.tsv), from `Shaders/*.bfsha.zst` and the effect and layout packages.
- Movies: 82, AV1 in WebM at 2560x1440 ([AV1 movies through SDK 22's decoder](../../../docs/reference/movies.md#av1-movies-through-sdk-22s-decoder)). A Mac without a hardware AV1 decoder (before M3) decodes them with dav1d.

## Shaders

All 16,888 translate (current). One shader has 10 texture handles the translator cannot resolve (`Failed to find handle source for bindless access of type "texture2D"`) and one logs `Invalid reduction type: F32FtzRn`; neither has been traced to anything visible (open).

## Rendering

- The battle map renders at 1024x576 and DLSS, on MetalFX here and FSR 2 on the Android host, takes it to 2048x1152 ([NVN2's DLSS on MetalFX](../../../docs/reference/rendering.md#nvn2s-dlss-on-metalfx), [NVN2's DLSS on FSR 2](../../../docs/reference/rendering.md#nvn2s-dlss-on-fsr-2-the-android-host)), which is scaled to the 3840x2160 window. The network's size follows the display ([The display's largest buffer](../../../docs/reference/services.md#the-displays-largest-buffer)).
- Auto exposure does not meter the scene. `5372c7e7cb7bbda2` writes one luminance into a 16x16 RGBA16F target (134.4 on the first map, log2 7.07), `6d0f6db3f0d396a3` makes a 256-bin histogram of it (the single bin 218), and `a0c632d8b5fa925b` averages its 10th to 90th percentile with a 64-bit division, takes the average through two 256x4 curves and moves a 3x1 R16F image towards the result. The tonemap reads `2^-(image[0] + 3) × 1.44`. On the first map `image` holds 8.96, 18.89, 9.46 with neither of the translator fixes below, and 5.555, 15.195, 6.055 with them, when the picture matches the footage. At a scene change the game clears the image to 1 and dispatches these shaders ([Pending clears land before a compute dispatch](../../../docs/reference/rendering.md#pending-clears-land-before-a-compute-dispatch)).
- Ambient occlusion is interleaved: depth into a 4-layer 512x288 array, four triangles picking the layer by primitive ID, then back together at 1024x576 ([`[[primitive_id]]` in a specialized Metal 4 pipeline](../../../docs/reference/rendering.md#primitive_id-in-a-specialized-metal-4-pipeline)).
- The enemy's checkered look and the pale squares over the ground are in the published footage too: both are the game's.
- The frame on the Android host: depth pre-pass (reverse Z, GEQUAL), G-buffer with depth EQUAL, deferred lighting, a forward pass with EQUAL for skinned meshes; FSR 2 takes 960x540 to 1920x1080. Any difference between a mesh's pre-pass and G-buffer positions leaves a black hole.
- Wind props (banners, awnings): pre-pass vertex shader `2365b68cdec1bbf4`, G-buffer `d04a126c1376d537`, instanced, a sine of time scaled by a sway weight at location 4. Black ribbons from them on the Android host are resolved ([Vertex attribute and stream binds replace the table](../../../docs/reference/rendering.md#vertex-attribute-and-stream-binds-replace-the-table-the-android-host)).
- Android host status (Odin 3, Adreno 830, Qualcomm's driver, 2026-11-10): the title screen, the floor and the plaza's props draw correctly. GPU about 12.7 ms of passes and compute a frame, FSR 2 about 3.5 ms; the largest single item is compute `9219893bed0b6ccd`.
- Frame time: 33 ms with the window on screen, and GPU about 12 ms. A covered window reads 80-180 ms, the host's display-link fallback (`front end: window covered`), which measures nothing.

## Driving it on the Mac

Keyboard in a window: J K U I are A B X Y, WASD the left stick, Shift + arrows the right stick, Z and C are ZL (threat range) and ZR (settings). The work folder has `win.sh` (a window with Metal's HUD, `run/ctl.txt` as `DM_CTL`), `mash.sh <secs> [dir]` (A every 0.8 s, with shots), `rebuild.sh` (a host copy, `build/fewhost`), `shaders.sh`, and `scripts/shdump.py <key> [ops]` for a shader's raw instructions. Pressing A from boot goes through the movie and the dialogue to the first battle map in about 80 s. Serials shift between runs, so read them from a pass dump; `DM_NVN_PASSDUMP_ALL=1` lists every dispatch's bindings.

## Open

- Nothing past the first map has been played on this build.
- `stage 1 cb4 has no uniform buffer bound`, once a run, with no effect seen.
- MetalFX's jitter sign and DLSS's bias mask are unchecked; the bias mask is only logged. No ghosting has been seen so far.
- FSR 2 on the Android host ([NVN2's DLSS on FSR 2](../../../docs/reference/rendering.md#nvn2s-dlss-on-fsr-2-the-android-host)) runs on the device; ghosting, the jitter's sign and the bias mask are unchecked.
- Android host: a pixel-shader set with corrected quad swaps (the work folder's `build/spirv_q2`) made the floor worse than the set in use (`spirv_sg`); why is not known.
- Android host: `an 8/16-bit integer texture read through a float declaration` is logged once a run (format 41), with no effect traced.
- Android host: the flat pink or magenta rectangles with the Turnip driver were dust particles drawn from vertex outputs the shader never writes ([Vertex outputs a shader does not write](../../../docs/reference/rendering.md#vertex-outputs-a-shader-does-not-write-the-android-host)); resolved in the host, both drivers now draw them the same.
- `dm/games/profile_guess.py` does not recognise the SDK's `Generic` version name.

## Found on this game

- [Scaled shared memory addresses](../../../docs/reference/shaders.md#scaled-shared-memory-addresses) and [IMAD.HI adds a register pair, and IMAD.X its carry](../../../docs/reference/shaders.md#imadhi-adds-a-register-pair-and-imadx-its-carry): the dark map.
- [Pending clears land before a compute dispatch](../../../docs/reference/rendering.md#pending-clears-land-before-a-compute-dispatch): battles and scene changes blown white.
- [NVN2 compute reads its workgroup counts at c0 + 4](../../../docs/reference/shaders.md#nvn2-compute-reads-its-workgroup-counts-at-c0--4).
- [Texture handles of SDK 22's Ampere code](../../../docs/reference/shaders.md#texture-handles-of-sdk-22s-ampere-code): white units.
- [A table of handles past the array's length](../../../docs/reference/shaders.md#a-table-of-handles-past-the-arrays-length): black characters.
- [An `else if` the SPIR-V writer never branched to](../../../docs/reference/shaders.md#an-else-if-the-spir-v-writer-never-branched-to): see-through characters on the Android host (resolved).
- [`[[primitive_id]]` in a specialized Metal 4 pipeline](../../../docs/reference/rendering.md#primitive_id-in-a-specialized-metal-4-pipeline): the dot lattice over the ground.
- [NVN2's DLSS on MetalFX](../../../docs/reference/rendering.md#nvn2s-dlss-on-metalfx).
- [AV1 movies through SDK 22's decoder](../../../docs/reference/movies.md#av1-movies-through-sdk-22s-decoder).
- [The Joy-Con mouse pointer](../../../docs/reference/services.md#the-joy-con-mouse-pointer) and [The display's largest buffer](../../../docs/reference/services.md#the-displays-largest-buffer).
- [An SDK 21 Switch 2 game with a 4.5 GB alias region](../../../docs/reference/memory.md#an-sdk-21-switch-2-game-with-a-45-gb-alias-region): SDK 22's node area.
- [Vertex attribute and stream binds replace the table](../../../docs/reference/rendering.md#vertex-attribute-and-stream-binds-replace-the-table-the-android-host): black ribbons from wind props on the Android host (resolved).
- Button prompts: the game draws them from its own picture font, `/Font/__nintendo_ext_003.bffnt.zst` with `_Large` and `_Small` (NintendoWare FFNT 4.1; one block-linear BC4 texture of glyph cells, stored bottom row first; 300 code points, the buttons at U+E0A0 to E0B6 outlined and U+E0E0 to E0F2 filled, the sticks at U+E0C1 / E0C2 and U+E101 / E102). It imports `nn::pl::GetBuiltInFontData` but never calls it (no shared-font line in any run's log), so the host's stand-in button font changes nothing. Prompts for another controller are a mod: the work folder's `re/fontmod/fontmod.py` paints icon pictures into those cells, re-encodes only the blocks it touched and writes the three files for `mods/<name>/romfs/Font/` (confirmed in the game on the Odin 3 with the PlayStation set, 2026-11-11). The mod derives from the game's file and is not shipped.
- [The early-depth kick wrote its pixel](../../../docs/reference/rendering.md#the-early-depth-kick-wrote-its-pixel-the-android-host): a dark patch over the world map on the Android host, the 1x1 shadow maps overwritten (resolved).
- [A depth clear waits for its image's next pass](../../../docs/reference/rendering.md#a-depth-clear-waits-for-its-images-next-pass-the-android-host) and [Uniform buffers off the device's offset alignment](../../../docs/reference/rendering.md#uniform-buffers-off-the-devices-offset-alignment-the-android-host): the floor missing in single frames on the Android host (resolved).
- [32-lane warps inside a wider host subgroup](../../../docs/reference/shaders.md#32-lane-warps-inside-a-wider-host-subgroup): the dark title screen and battles on the Android host (resolved).
- [Integer vertex inputs declared float](../../../docs/reference/shaders.md#integer-vertex-inputs-declared-float-the-vulkan-chain) and [The Android host's rolling capture](../../../docs/reference/rendering.md#the-android-hosts-rolling-capture).
