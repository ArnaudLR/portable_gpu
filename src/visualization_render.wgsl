// ===========================================================================
//  visualization_render.wgsl — affichage à l'écran de la texture de projection
// ===========================================================================
//
//  ★ COMMENT LIRE CE FICHIER ★
//    • les lignes préfixées par [WGSL] expliquent le LANGAGE WGSL lui-même
//      (annotations « @ », types, builtins, API WebGPU correspondante).
//    • les autres commentaires expliquent CE QUE FAIT le code et POURQUOI.
//
//  C'est le seul fichier GRAPHIQUE du projet (les deux autres sont des compute
//  shaders) et il contient les deux points d'entrée d'une pipeline de rendu :
//    • vertex_main   : fabrique un unique triangle qui recouvre tout l'écran
//                      (aucun buffer de sommets n'est nécessaire) ;
//    • fragment_main : lit la texture 64 × 32 écrite par visualization.wgsl et
//                      en écrit les couleurs dans la surface de la fenêtre.
//
//  Ce shader est volontairement TRÈS simple : pas de tableau constant, pas
//  d'indexation dynamique, pas de sampler, pas d'échantillonnage avec
//  filtrage. Les compilateurs D3D12/Intel anciens sur lesquels le projet a été
//  testé supportent mal les shaders « modernes » : la simplicité est ici une
//  exigence de portabilité, pas une paresse. (C'est le même souci de robustesse
//  qui fait choisir explicitement le format de surface dans src/lbm.cpp.)
//  Les dimensions 64 × 32 sont celles de kWidth / kHeight dans src/lbm.cpp.
// ===========================================================================

// [WGSL] Une texture LUE par un shader est une « sampled texture » : ici une
// texture 2D dont les texels sont lus comme des f32.
// [WGSL] @group(0)/@binding(0) : mêmes numéros que le bind group créé par
// l'hôte (textureEntry / textureGroup dans src/lbm.cpp).
// [WGSL] Aucun sampler n'est déclaré, et c'est délibéré : on n'utilise que
// textureLoad (lecture directe d'un texel par coordonnée entière), qui ne
// filtre pas et n'a donc pas besoin de sampler. C'est plus simple, parfaitement
// déterministe, et adapté à un agrandissement « façon pixel art ».
@group(0) @binding(0) var image: texture_2d<f32>;

// [WGSL] Cette structure est une INTERFACE ENTRE ÉTAPES : c'est la sortie de
// vertex_main et l'entrée de fragment_main. Le GPU interpole les valeurs
// marquées @location entre les trois sommets du triangle (c'est ainsi que uv
// varie en douceur sur toute la surface), tandis que @builtin(position) est
// traitée à part par le rastériseur.
// [WGSL] @builtin(position) est la position en espace de découpage (clip
// space) : c'est la seule sortie qu'une étape de sommets DOIT fournir. Les
// coordonnées vont de −1 à +1, l'axe y pointant vers le haut (convention
// D3D/Metal adoptée par WebGPU, contrairement à Vulkan), tandis que les
// coordonnées de la fenêtre, elles, ont leur origine en haut à gauche et un
// axe y descendant.
// [WGSL] @location(0) est une interface libre : aucun sens n'est imposé, le
// seul impératif est que le numéro corresponde entre le vertex shader et le
// fragment shader.
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// [WGSL] @vertex marque le point d'entrée de l'étape « sommets » ; le nom est
// référencé en C++ (renderDesc.vertex.entryPoint = "vertex_main").
// [WGSL] @builtin(vertex_index) est le numéro du sommet dans l'appel de
// dessin : le pipeline exécute cette fonction 3 fois (draw(3, 1, 0, 0) dans
// src/lbm.cpp) avec index = 0, 1 puis 2. C'est la seule information nécessaire :
// les positions en découlent par calcul, donc ni buffer de sommets ni attribut
// n'est requis.
// Le triangle couvre tout l'écran (technique du « full-screen triangle ») :
// ses trois sommets tombent en dehors de la fenêtre, mais le triangle contient
// tout le carré [−1,1]² — un seul triangle suffit donc là où deux triangles
// laisseraient une couture diagonale au milieu de l'image.
@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOut {
    // Triangle plein écran obtenu par arithmétique pure, sans tableau constant
    // ni indexation dynamique : c'est exactement le genre de construction que
    // les vieux compilateurs D3D12/Intel gèrent sans broncher.
    // [WGSL] << et & sont les opérateurs bit à bit (décalage à gauche et ET) :
    // c'est ici une astuce sans tableau ni « if » qui donne, selon index :
    //      index = 0 -> x = 0, y = 0
    //      index = 1 -> x = 2, y = 0
    //      index = 2 -> x = 0, y = 2
    // Le suffixe « u » (1u, 2u) dénote des littéraux u32 ; f32(...) convertit
    // explicitement le résultat en flottant (WGSL n'a aucune conversion
    // implicite entre entiers et flottants).
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    // [WGSL] var = variable locale modifiable : on la remplit champ par champ
    // avant de la retourner. Une structure doit être construite membre à
    // membre, il n'existe pas d'initialisation « en bloc » en WGSL.
    var output: VertexOut;
    // Passage des coordonnées (0,0), (2,0), (0,2) vers l'espace de découpage :
    // x*2−1 donne −1, 3, −1 et 1−y*2 donne +1, +1, −3. Le sommet 0 devient
    // donc le coin haut-gauche (−1, +1) de la fenêtre, et le triangle couvre
    // tout le reste en débordant très largement sur la droite et le bas.
    // Le « z = 0 » est la profondeur minimale (0 = proche, 1 = lointain en
    // WebGPU) et le « 1 » final de w annule toute perspective : la position est
    // déjà en coordonnées normalisées.
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    // uv réutilise les mêmes x et y, qui valent 0, 1 ou 2 : les valeurs 2
    // sortent de l'intervalle [0,1], ce qui est sans conséquence puisque le
    // fragment shader borne uv avant de s'en servir (voir plus bas). Le sommet
    // 0 a bien uv = (0,0) au coin haut-gauche de la fenêtre, ce qui correspond
    // au texel (0,0) de la texture : l'image n'est donc ni retournée ni
    // décalée, la convention « origine en haut à gauche » étant la même des
    // deux côtés.
    output.uv = vec2<f32>(x, y);
    return output;
}

// [WGSL] @fragment marque le point d'entrée de l'étape « fragments » : cette
// fonction est appelée une fois par pixel couvert par le triangle (donc pour
// chaque pixel de la fenêtre). Le type de retour est annoté @location(0) : la
// valeur produite alimente la cible de couleur n°0, celle qui correspond au
// format de la surface (BGRA8Unorm sur la plupart des configurations, voir la
// sélection de surfaceFormat dans src/lbm.cpp). Les composantes flottantes sont
// converties automatiquement vers ce format.
// Le paramètre « input » est la structure interpolée produite par vertex_main.
@fragment fn fragment_main(input: VertexOut) -> @location(0) vec4<f32> {
    // [WGSL] clamp(x, lo, hi) borne chaque composante (ici, vec2<f32>(0.0) est
    // un vecteur dont les composantes valent 0 : c'est la façon d'écrire un
    // « splat » en WGSL).
    // Borner à 0.999 au lieu de 1.0 est indispensable : les coordonnées texel
    // vont de 0 à 63 en x et de 0 à 31 en y, et la troncature de uv*64 ou
    // uv*32 sur un uv exactement égal à 1.0 donnerait 64 ou 32, donc un accès
    // hors de la texture. Le compromis choisi (rétrécir d'un millième de
    // texel) est invisible et évite tout cas limite.
    let uv = clamp(input.uv, vec2<f32>(0.0), vec2<f32>(0.999));
    // [WGSL] vec2<i32>(v) convertit un vec2<f32> en entiers en tronquant vers
    // zéro : uv * (64, 32) donne donc directement la coordonnée du texel à
    // lire. Autrement dit, la texture 64 × 32 est agrandie en « plus proche
    // voisin » (chaque texel devient un gros bloc de pixels à l'écran) — le
    // rendu volontairement simple de la visualisation.
    let pixel = vec2<i32>(uv * vec2<f32>(64.0, 32.0));
    // [WGSL] textureLoad(texture, coordonnée, niveau_de_mip) : lecture brute
    // d'un texel, sans sampler, sans filtrage, sans coordonnées normalisées.
    // C'est l'opération symétrique de textureStore() dans visualization.wgsl,
    // et elle n'échantillonne ici que le niveau 0 (cette texture n'a pas de
    // mipmaps : mipLevelCount = 1 côté CPU).
    return textureLoad(image, pixel, 0);
}
