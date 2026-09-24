// ===========================================================================
//  visualization.wgsl — projection du champ D3Q19 en une image RGBA 64 × 32
// ===========================================================================
//
//  ★ COMMENT LIRE CE FICHIER ★
//    • les lignes préfixées par [WGSL] expliquent le LANGAGE WGSL lui-même
//      (annotations « @ », types, builtins, API WebGPU correspondante).
//    • les autres commentaires expliquent CE QUE FAIT le code et POURQUOI.
//
//  ── 1) RÔLE DANS L'APPLICATION ───────────────────────────────────────────
//  Ce compute shader tourne après chaque pas de temps LBM (voir la boucle de
//  src/lbm.cpp) et convertit le gros buffer de populations
//  (64 × 32 × 32 nœuds × 19 f32 ≈ 5 Mo) en une petite texture 64 × 32 pixels.
//  La passe de rendu finale (visualization_render.wgsl) ne fait ensuite que
//  recopier cette texture à l'écran.
//
//  Pourquoi cette étape intermédiaire ? Parce que lire les populations exige un
//  accès à un storage buffer, impossible dans une étape de rendu classique.
//  La solution « naturelle » serait de tout faire dans le fragment shader, mais
//  la compilation d'un tel shader plante sur certains pilotes Intel/D3D12
//  anciens (c'est le même genre de fragilité que la sélection de format de
//  surface dans lbm.cpp). On déporte donc tout le calcul lourd dans un compute
//  shader, et la passe graphique se contente d'une lecture de texture.
//
//  ── 2) CE QUE L'ON VOIT ──────────────────────────────────────────────────
//  Une projection de « volume » volontairement naïve le long de l'axe z : pour
//  chaque pixel (x, y) de l'image, on parcourt en profondeur les 32 nœuds de la
//  colonne et on cumule la lumière émise par ceux dont la vitesse est non nulle
//  (en partant de z = 0, qui joue le rôle de « premier plan »). Les zones
//  lentes apparaissent bleues, les zones rapides orangées, et la sphère solide
//  grise. C'est un rendu esthétique, pas une mesure physique : l'objectif est
//  de rendre l'écoulement lisible, et cette approximation suffit.
// ===========================================================================

// [WGSL] Même structure — et donc même disposition mémoire — que dans
// lbm_kernels.wgsl et que le struct Parameters de src/lbm.cpp : les définitions
// doivent coïncider OCTET PAR OCTET, sinon chaque invocation lit des nombres
// issus de la réinterprétation de bits voisins.
// [WGSL] Les règles d'alignement de l'espace « uniform » imposent ici que la
// taille totale fasse un multiple de 16 octets (d'où le 4e u32 réservé dans
// dimensions et le champ _padding) et qu'un vec4 soit aligné sur 16 octets.
struct Parameters {
    dimensions: vec4<u32>, omega: f32, initial_velocity: f32,
    obstacle_radius: f32, _padding: f32,
};

// [WGSL] @group(0)/@binding(0..2) : mêmes numéros que ceux déclarés côté hôte
// dans src/lbm.cpp (voir projectEntries et wgpuComputePassEncoderSetBindGroup).
// [WGSL] var<storage, read> = buffer de stockage en lecture seule. C'est le
// buffer de populations que le kernel LBM vient d'écrire : le même binding
// change de buffer à chaque pas selon le ping-pong (projectGroupA /
// projectGroupB côté C++).
// [WGSL] texture_storage_2d<rgba8unorm, write> = texture 2D accessible en
// ÉCRITURE depuis un shader. On ne peut ni la lire ni la filtrer : son seul
// accès possible est textureStore, et le format « rgba8unorm » (4 octets par
// texel, valeurs [0,1] quantifiées sur 8 bits) doit correspondre exactement à
// la texture créée par le CPU (WGPUTextureFormat_RGBA8Unorm et
// WGPUStorageTextureAccess_WriteOnly dans lbm.cpp).
@group(0) @binding(0) var<storage, read> populations: array<f32>;
@group(0) @binding(1) var<uniform> params: Parameters;
@group(0) @binding(2) var projection: texture_storage_2d<rgba8unorm, write>;

// Même table de directions D3Q19 que DIRECTIONS dans lbm_kernels.wgsl et
// kLatticeDirections en C++, mais en vec3<f32> : ici les vecteurs servent à
// pondérer des populations dans un calcul de moyenne, pas à indexer un voisin.
// Seule la cohérence des valeurs importe — l'ordre des paires doit rester
// identique pour que la vitesse calculée ait le bon signe.
const D: array<vec3<f32>, 19> = array<vec3<f32>, 19>(
 vec3<f32>(0,0,0),vec3<f32>(1,0,0),vec3<f32>(-1,0,0),vec3<f32>(0,1,0),vec3<f32>(0,-1,0),
 vec3<f32>(0,0,1),vec3<f32>(0,0,-1),vec3<f32>(1,1,0),vec3<f32>(-1,-1,0),vec3<f32>(1,-1,0),
 vec3<f32>(-1,1,0),vec3<f32>(1,0,1),vec3<f32>(-1,0,-1),vec3<f32>(1,0,-1),vec3<f32>(-1,0,1),
 vec3<f32>(0,1,1),vec3<f32>(0,-1,-1),vec3<f32>(0,1,-1),vec3<f32>(0,-1,1));

// PHYSIQUE — reconstruit la vitesse macroscopique d'un nœud à partir de ses 19
// populations, exactement comme dans la partie « grandeurs macroscopiques » de
// simulate() :
//      ρ = Σ f_i        u = (Σ f_i · c_i) / ρ
// Le code est dupliqué ici parce qu'un module WGSL ne peut pas appeler une
// fonction définie dans un autre module : chaque fichier .wgsl forme un module
// autonome, compilé séparément par le CPU. Dupliquer ces trois lignes coûte
// moins cher que de faire transiter ρ et u par un buffer intermédiaire — et ce
// calcul ne sert qu'à COLORER l'image, il n'a pas à être aussi soigné que dans
// le kernel de simulation, dont dépend la solution physique.
//
// [WGSL] Une fonction ne renvoie qu'une valeur, mais ici seul u nous intéresse :
// ρ est recalculé localement puis jeté. [WGSL] var = variable modifiable,
// let = valeur liée une fois ; les espaces et retours à la ligne du corps sont
// purement cosmétiques (tout le corps tient sur une ligne).
// [WGSL] max() évite la division par zéro : un nœud de densité 0 donnerait
// 0/0 = NaN, et un seul NaN suffirait à rendre tout le rendu incohérent
// (NaN se propage à travers les comparaisons et les mélanges de couleurs).
fn velocity(site: u32) -> vec3<f32> {
 var rho=0.0; var momentum=vec3<f32>(0.0);
 for(var i=0u;i<19u;i+=1u){let f=populations[site*19u+i];rho+=f;momentum+=f*D[i];}
 return momentum/max(rho,0.0001);
}

// [WGSL] @compute + @workgroup_size(8,8) : groupes de 64 invocations disposées
// en 2D — un découpage naturel pour une image. src/lbm.cpp dispatche
// ((64+7)/8) × ((32+7)/8) × 1 = 8 × 4 groupes, soit une invocation par pixel.
// [WGSL] global_invocation_id vaut ici exactement le pixel (id.x, id.y).
@compute @workgroup_size(8,8)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
 // [WGSL] .xyz est un « swizzle » : on extrait les trois premières composantes
 // du vec4 (la 4e est le champ réservé). Il donne donc un vec3<u32>.
 let dims=params.dimensions.xyz;
 // Le dispatch est arrondi au multiple de 8 supérieur : si dims n'était pas un
 // multiple de 8, des invocations déborderaient de l'image et écriraient hors
 // de la texture (ce qui est refusé par la validation WebGPU). Ici 64 et 32
 // sont des multiples de 8, mais la garde rend le shader générique — et une
 // écriture hors bornes dans un storage buffer est une erreur, pas un détail.
 if(id.x>=dims.x || id.y>=dims.y){return;}
 // Couleur de fond (bleu nuit) et opacité accumulée : ce sont les deux
 // accumulateurs du mélange alpha réalisé dans la boucle ci-dessous. Aucun
 // « clear » n'est nécessaire, la valeur de départ est ici.
 var colour=vec3<f32>(0.008,0.012,0.025); var alpha=0.0;
 // Balayage en profondeur : z = 0 à dims.z − 1, c'est ce qui transforme un
 // volume 3D en une image 2D. La « caméra » est placée du côté des petits z,
 // la première couche traversée est donc la plus proche de l'observateur.
 for(var z=0u;z<dims.z;z+=1u){
  // Même indexation que site_index() de lbm_kernels.wgsl : x le plus rapide,
  // puis y, puis z, avec 19 flottants consécutifs par nœud.
  let site=(z*dims.y+id.y)*dims.x+id.x;
  // [WGSL] length(vec) = norme euclidienne, calculée par le GPU (équivalent de
  // sqrt(dot(v, v))). speed est la vitesse du fluide en unités de réseau :
  // 0 au repos, ≈ 0.04 dans l'écoulement d'entrée, davantage en contournant la
  // sphère.
  let speed=length(velocity(site));
  // [WGSL] smoothstep(e0, e1, x) renvoie 0 pour x ≤ e0, 1 pour x ≥ e1, et une
  // interpolation en S (dérivée continue) entre les deux. Ici : les nœuds
  // quasi immobiles (speed < 0.005, donc au repos) ne contribuent pas du tout
  // au rendu, et l'intensité monte progressivement jusqu'à 0.075. Le facteur
  // 0.10 rend chaque nœud très transparent — c'est la somme de 32 nœuds qui
  // dessine le volume. Sans ce seuil, tout le domaine brillerait uniformément.
  let opacity=smoothstep(0.005,0.075,speed)*0.10;
  // [WGSL] mix(a, b, t) interpole linéairement (a·(1−t) + b·t) et clamp(x, lo,
  // hi) borne x. La teinte va donc du bleu (fluide lent) à l'orange (fluide
  // rapide), la normalisation par 0.09 faisant saturer la couleur dès
  // 0.09 — bien au-delà de la vitesse d'entrée de 0.04, ce qui laisse de la
  // marge pour visualiser les accélérations autour de l'obstacle.
  let mapped=mix(vec3<f32>(0.03,0.25,0.85),vec3<f32>(1.0,0.35,0.03),clamp(speed/0.09,0.0,1.0));
  // Composition alpha « de l'avant vers l'arrière » : chaque couche ne
  // contribue qu'à travers la transparence restante (1 − alpha), puis l'opacité
  // cumulée augmente de la même façon. C'est la formule classique
  //     C += (1 − A) · α · c        A += (1 − A) · α
  // Les premières couches traversées (les z les plus petits) dominent donc
  // naturellement, sans qu'il soit besoin de trier ni de reculer un quelconque
  // plan de coupe : le volume s'estompe à mesure qu'on s'enfonce.
  colour+=(1.0-alpha)*opacity*mapped; alpha+=(1.0-alpha)*opacity;
 }
 // La sphère solide, vue de face, est un disque de même rayon que dans
 // is_solid() : son centre (x, y) est celui de la sphère 3D, la projection le
 // long de z ne changeant ni x ni y. On le grise pour le distinguer du fluide.
 // [WGSL] distance(a, b) = length(a − b) ; vec2<f32>(id.xy) reconvertit un
 // vec2<u32> en flottants. On tire la couleur accumulée à 65 % vers un gris
 // moyen, quel que soit le fluide qui se trouve devant ou derrière l'obstacle.
 let center=vec2<f32>(f32(dims.x)*0.25,f32(dims.y)*0.5);
 if(distance(vec2<f32>(id.xy),center)<params.obstacle_radius){colour=mix(colour,vec3<f32>(0.65),0.65);}
 // [WGSL] textureStore écrit un texel par ses coordonnées ENTIÈRES : pas de
 // sampler, pas de coordonnées normalisées, pas de filtrage. L'écriture
 // remplace le texel (aucun mélange), les valeurs flottantes sont bornées à
 // [0,1] puis quantifiées sur 8 bits par le format rgba8unorm.
 // L'alpha est fixé à 1.0 : le canal alpha de cette texture n'est jamais
 // utilisé par la suite (la passe de rendu ne lit que RVB), il ne doit pas
 // être confondu avec l'accumulateur alpha de la boucle ci-dessus.
 textureStore(projection,vec2<i32>(id.xy),vec4<f32>(colour,1.0));
}
