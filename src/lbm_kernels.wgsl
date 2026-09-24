// ===========================================================================
//  lbm_kernels.wgsl — un pas de temps « lattice Boltzmann » D3Q19 (collision BGK)
// ===========================================================================
//
//  ★ COMMENT LIRE CE FICHIER ★
//    • les lignes préfixées par [WGSL] expliquent le LANGAGE WGSL lui-même :
//      ses annotations « @ », ses types, ses fonctions natives (builtins) et
//      l'API WebGPU à laquelle tout cela correspond (bind groups, buffers...).
//    • les autres commentaires expliquent CE QUE FAIT le code et POURQUOI il
//      est écrit ainsi (physique, algorithme, performance, portabilité).
//
//  ── 1) CE QUI EST SIMULÉ ─────────────────────────────────────────────────
//  On simule un fluide incompressible dans une grille de 64 × 32 × 32 mailles
//  (constantes kWidth / kHeight / kDepth de src/lbm.cpp).
//  La méthode « lattice Boltzmann » (LBM) ne résout pas directement les
//  équations de Navier-Stokes. Elle stocke, en chaque nœud du réseau et pour
//  19 directions de propagation (le modèle « D3Q19 »), la quantité de matière
//  qui se déplace dans cette direction : ce sont les 19 « populations » f_i.
//
//  Toutes les grandeurs sont exprimées en UNITÉS DE RÉSEAU (Δx = Δt = 1) : les
//  vitesses sont donc sans dimension. La vitesse d'entrée de 0.04 signifie
//  « 0.04 maille par pas de temps », ce qui est très inférieur à la vitesse du
//  son du réseau (1/√3 ≈ 0.577) : c'est la condition de faible nombre de Mach
//  (≈ 0.07) qui rend ce modèle valable pour un fluide incompressible.
//
//  Un pas de temps se décompose en deux étapes purement locales :
//    1. STREAMING  : chaque population se déplace d'un nœud vers le nœud
//                    voisin situé dans sa direction de propagation.
//    2. COLLISION  : à chaque nœud, les 19 populations sont ramenées vers
//                    leur équilibre local (schéma BGK, un seul paramètre ω),
//                    ce qui traduit les effets de pression et de viscosité.
//  C'est cette alternance, répétée des milliers de fois, qui fait émerger
//  l'équation de Navier-Stokes incompressible à grande échelle : les
//  écoulements visibles à l'écran (sillage derrière la sphère, zones
//  accélérées sur ses flancs) sont un comportement émergent de ces deux
//  étapes microscopiques.
//
//  ── 2) MODÈLE D'EXÉCUTION ────────────────────────────────────────────────
//  Ce fichier est un COMPUTE SHADER : il ne dessine rien, il calcule. Le GPU
//  exécute des milliers de copies du point d'entrée « simulate » en
//  parallèle, une par nœud du réseau. Chacune lit populations_in et écrit
//  populations_out ; ni verrou ni barrière n'est nécessaire (voir « pull »
//  plus bas).
//
//  src/lbm.cpp crée le module shader (wgpuDeviceCreateShaderModule), la
//  pipeline de calcul puis les bind groups, et lance un dispatch de
//  simulate() par image affichée, en alternant deux buffers (« ping-pong ») :
//
//      pas 0 : populationsA -> populationsB      (bindGroupAB)
//      pas 1 : populationsB -> populationsA      (bindGroupBA)
//      pas 2 : populationsA -> populationsB      ...
//
//  Ce ping-pong est indispensable : lire et écrire le MÊME buffer pendant un
//  dispatch serait une course de données (une invocation lirait une valeur
//  déjà réécrite par sa voisine), et le résultat dépendrait alors de l'ordre
//  d'exécution — donc du GPU, donc non reproductible.
// ===========================================================================

// [WGSL] struct = description de la disposition OCTET PAR OCTET d'un bloc de
// données. Ce n'est pas juste un type : c'est un contrat d'interface avec le
// CPU. Le même struct est défini en C++ dans src/lbm.cpp
// (static_assert(sizeof(Parameters) == 32)) : si les deux définitions
// divergent d'un seul octet, le shader lit des valeurs absurdes (ω = 3.7e-41…)
// et la simulation explose silencieusement.
// [WGSL] Règles d'alignement : dans l'espace d'adressage « uniform », un
// vec4<u32> est aligné sur 16 octets et la taille totale d'une structure
// partagée avec le CPU doit aussi être un multiple de 16. C'est le rôle du
// « réservé » de dimensions (4e composante du vec4) et du champ _padding :
// ils donnent à la structure la même disposition que celle du C++
// (std::array<std::uint32_t, 4> occupe 16 octets), soit 32 octets au total.
// Avec un vec3<u32> nu, la structure ferait 28 octets et le module serait
// purement et simplement refusé à la validation.
struct Parameters {
    dimensions: vec4<u32>, // x, y, z, réservé (complète l'alignement de 16 octets)
    omega: f32,
    initial_velocity: f32,
    obstacle_radius: f32,
    _padding: f32,
};

// [WGSL] Les ressources du shader (buffers, textures, samplers) se déclarent
// au niveau module, avec une annotation @group(n) @binding(m). Ce couple
// (numéro de groupe, numéro de binding) est le seul lien avec le CPU : c'est
// lui qu'on retrouve dans les WGPUBindGroupLayoutEntry de src/lbm.cpp, puis
// dans le bind group qui fournit les buffers concrets au moment du dispatch.
//
// [WGSL] var<storage> déclare un accès à la mémoire globale du GPU. Deux modes
// d'accès seulement : read (lecture seule) et read_write. Il n'existe pas de
// mode « write only » : le buffer de sortie est donc déclaré read_write même
// s'il n'est jamais relu ici — c'est ce que déclare aussi le CPU
// (WGPUBufferBindingType_Storage côté sortie, ReadOnlyStorage côté entrée).
//
// [WGSL] array<f32> sans taille = tableau de longueur dynamique : sa taille
// réelle est celle du buffer lié au binding. Une seule dimension, plate :
// c'est le shader qui calcule l'indice de chaque population (voir site_index).
@group(0) @binding(0) var<storage, read> populations_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> populations_out: array<f32>;
@group(0) @binding(2) var<uniform> params: Parameters;

// [WGSL] const = constante évaluée à la compilation. Comme les indices de
// boucle utilisés plus bas sont bornés et constants, le compilateur peut
// « dérouler » la boucle : les 19 directions et les 19 poids finissent empilés
// dans des registres, sans le moindre accès à la mémoire globale.
// [WGSL] vec3<i32> = vecteur de 3 entiers SIGNÉS (les directions valent -1, 0
// ou 1). Le constructeur explicite array<vec3<i32>, 19>(...) est obligatoire :
// il n'y a pas d'initialisation implicite par accolades en WGSL.
//
// PHYSIQUE — les 19 directions du modèle D3Q19, classées par longueur :
//   i = 0        : repos                            |c|² = 0
//   i = 1 à 6    : voisins de FACE                 |c|² = 1
//   i = 7 à 18   : diagonales d'ARÊTE             |c|² = 2
// L'ordre est un contrat partagé : il doit être identique au tableau
// kLatticeDirections de src/lbm.cpp (pour l'état initial) et au tableau D de
// visualization.wgsl (pour la mesure des vitesses), sinon WEIGHTS[i] et
// OPPOSITE[i] ne correspondraient plus aux bonnes directions.
// Les paires (1,2), (3,4) … (17,18) sont deux à deux opposées : c'est ce qui
// rend la table OPPOSITE ci-dessous presque triviale à écrire.
const DIRECTIONS: array<vec3<i32>, 19> = array<vec3<i32>, 19>(
    vec3<i32>( 0, 0, 0),
    vec3<i32>( 1, 0, 0), vec3<i32>(-1, 0, 0),
    vec3<i32>( 0, 1, 0), vec3<i32>( 0,-1, 0),
    vec3<i32>( 0, 0, 1), vec3<i32>( 0, 0,-1),
    vec3<i32>( 1, 1, 0), vec3<i32>(-1,-1, 0),
    vec3<i32>( 1,-1, 0), vec3<i32>(-1, 1, 0),
    vec3<i32>( 1, 0, 1), vec3<i32>(-1, 0,-1),
    vec3<i32>( 1, 0,-1), vec3<i32>(-1, 0, 1),
    vec3<i32>( 0, 1, 1), vec3<i32>( 0,-1,-1),
    vec3<i32>( 0, 1,-1), vec3<i32>( 0,-1, 1)
);

// PHYSIQUE — les poids w_i de la quadrature D3Q19 : 1/3 pour la direction
// immobile, 1/18 pour chacune des 6 directions de face, 1/36 pour chacune des
// 12 diagonales. Ces valeurs proviennent du développement de la distribution
// de Maxwell-Boltzmann sur les 19 directions ; leur somme vaut exactement 1,
// ce qui garantit qu'un équilibre de densité ρ transporte bien une masse ρ.
// Les divisions « 1.0 / 3.0 » sont repliées en constantes par le compilateur.
const WEIGHTS: array<f32, 19> = array<f32, 19>(
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);

// PHYSIQUE — table des directions opposées : OPPOSITE[i] est l'indice de la
// direction -c_i. Elle sert au « bounce-back » : une population qui devrait
// entrer dans une paroi est remplacée par celle qui en sort, retournée de
// 180°, ce qui annule la vitesse normale à la paroi.
// [WGSL] Le suffixe « u » (0u, 2u…) dénote un littéral u32 : le type de
// l'élément du tableau est array<u32, 19>, à ne pas confondre avec les i32 de
// DIRECTIONS. WGSL n'a aucune conversion implicite entre types numériques.
const OPPOSITE: array<u32, 19> = array<u32, 19>(
    0u, 2u, 1u, 4u, 3u, 6u, 5u,
    8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u,
    16u, 15u, 18u, 17u
);

// [WGSL] fn déclare une fonction ; les paramètres sont typés et passés par
// valeur, la valeur de retour suit « -> ».
//
// UTILITAIRE — convertit une coordonnée 3D du réseau en indice linéaire,
// formule identique à celle du CPU : site = (z * Ny + y) * Nx + x.
// Autrement dit, x est l'indice le plus rapide, puis y, puis z. Chaque site
// possède 19 populations consécutives rangées dans l'ordre des directions :
// la population i du site s se lit donc dans populations[s * 19 + i].
// Ce stockage « 19 valeurs par nœud » est volontaire : les 19 lectures d'un
// nœud sont contiguës en mémoire, ce qui améliore la coalescence des accès
// (le GPU regroupe les lectures voisines des invocations d'un même groupe).
fn site_index(x: u32, y: u32, z: u32, dims: vec3<u32>) -> u32 {
    return (z * dims.y + y) * dims.x + x;
}

// GÉOMÉTRIE — indique si le nœud « position » est solide. Deux cas :
//   • position.y == 0 ou position.y + 1 == dims.y : les deux parois
//     horizontales du canal. Les directions x et z restent PÉRIODIQUES
//     (le domaine est un tunnel sans fin, couplé à un vent d'entrée).
//   • à l'intérieur de la sphère de rayon params.obstacle_radius centrée au
//     quart de la longueur du domaine, à mi-hauteur et mi-profondeur.
// Cette fonction doit rester le miroir exact de la boucle d'initialisation du
// CPU (make_initial_populations dans src/lbm.cpp) : si une seule maille est
// classée différemment, le premier pas de temps démarre sur un état
// incohérent (des populations à l'intérieur d'un solide, par exemple).
//
// [WGSL] f32(...) et vec3<f32>(...) sont des CONVERSIONS explicites : en WGSL
// il n'y a aucune conversion implicite entre u32, i32 et f32. f32(position)
// convertit les trois composantes d'un seul coup.
// [WGSL] L'opérateur == sur les vecteurs compare composante par composante et
// le || s'applique au résultat booléen.
// [WGSL] dot(a, b) est le produit scalaire : comparer des distances au carré
// évite un sqrt() par test, ce qui compte puisque cette fonction est appelée
// jusqu'à 19 fois par nœud et par pas de temps.
fn is_solid(position: vec3<u32>, dims: vec3<u32>) -> bool {
    if (position.y == 0u || position.y + 1u == dims.y) {
        return true;
    }
    let center = vec3<f32>(f32(dims.x) * 0.25, f32(dims.y) * 0.5, f32(dims.z) * 0.5);
    let offset = vec3<f32>(position) - center;
    return dot(offset, offset) <= params.obstacle_radius * params.obstacle_radius;
}

// PHYSIQUE — équilibre de Maxwell-Boltzmann tronqué au 2e ordre en vitesse et
// discrétisé sur les 19 directions : c'est le f_i^eq du schéma BGK.
//
//      f_i^eq = w_i · ρ · ( 1 + 3 (c_i · u) + 4.5 (c_i · u)² − 1.5 |u|² )
//
// Les coefficients 3, 4.5 et 1.5 ne sont pas des réglages : ils découlent de la
// vitesse du son du réseau (c_s² = 1/3), avec 1/c_s² = 3, 1/(2 c_s⁴) = 4.5 et
// 1/(2 c_s²) = 1.5. C'est ce développement qui, moyennant une analyse de
// Chapman-Enskog, fait réapparaître les équations d'Euler puis de Navier-Stokes
// incompressible à faible nombre de Mach : le GPU ne résout donc jamais
// Navier-Stokes explicitement, il ne fait que streamer et relaxer des
// populations.
//
// [WGSL] DIRECTIONS[direction] est un vec3<i32> : la conversion vers
// vec3<f32> doit être explicite. Les opérateurs * et + s'appliquent
// composante par composante (ce n'est pas une multiplication matricielle).
fn equilibrium(direction: u32, density: f32, velocity: vec3<f32>) -> f32 {
    let c_dot_u = dot(vec3<f32>(DIRECTIONS[direction]), velocity);
    let u_squared = dot(velocity, velocity);
    return WEIGHTS[direction] * density *
        (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u * c_dot_u - 1.5 * u_squared);
}

// [WGSL] @compute marque le point d'entrée d'un compute shader. Le nom
// « simulate » est celui référencé en C++ (computeStage.entryPoint = "simulate")
// : un module peut contenir plusieurs points d'entrée, c'est le CPU qui choisit
// lequel exécuter.
// [WGSL] @workgroup_size(4,4,4) définit la taille d'un GROUPE DE TRAVAIL :
// 64 invocations lancées ensemble sur un même cœur du GPU, qui peuvent
// partager de la mémoire (var<workgroup>) et se synchroniser
// (workgroupBarrier). Ici aucun partage n'est utilisé : chaque invocation est
// totalement indépendante, ce qui la rend « embarrassingly parallel » et
// parfaitement adaptée au GPU. src/lbm.cpp dispatche
// (64/4) × (32/4) × (32/4) = 16 × 8 × 8 groupes, soit exactement un fil
// d'exécution par nœud du réseau. 4×4×4 = 64 invocations reste bien en dessous
// des limites garanties par WebGPU (maxComputeInvocationsPerWorkgroup = 256).
// [WGSL] @builtin(global_invocation_id) est une valeur fournie par le GPU et
// non par le code : c'est l'identifiant GLOBAL de l'invocation dans la grille
// de dispatch, ici directement les coordonnées (x, y, z) du nœud traité.
// [WGSL] Une fonction sans valeur de retour n'a pas de « -> » et s'arrête avec
// un « return » nu.
@compute @workgroup_size(4, 4, 4)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = params.dimensions.xyz;
    if (id.x >= dims.x || id.y >= dims.y || id.z >= dims.z) {
        return;
    }

    let site = site_index(id.x, id.y, id.z, dims);
    let position = id;
    if (is_solid(position, dims)) {
        // Nœud solide : la matière n'y circule pas, et le « pull » ci-dessous
        // ne lira JAMAIS un nœud solide (le bounce-back s'en charge) — ce
        // contenu n'influence donc pas la physique.
        // On l'écrit malgré tout, avec une valeur déterministe : l'équilibre au
        // repos (ρ = 1, u = 0), c'est-à-dire l'état vers lequel tend un
        // bounce-back parfait. Deux raisons : le buffer de sortie devient le
        // buffer d'entrée du pas suivant, il ne doit donc contenir que des
        // valeurs que l'on sait relire ; et la visualisation, elle, parcourt
        // TOUTE la colonne, solides compris — une valeur aberrante y ferait
        // apparaître du bruit, et un seul NaN y contaminerait une ligne entière
        // de l'image.
        for (var i = 0u; i < 19u; i += 1u) {
            populations_out[site * 19u + i] = equilibrium(i, 1.0, vec3<f32>(0.0));
        }
        return;
    }

    // [WGSL] var = variable modifiable (ici, locale à la fonction, donc dans
    // l'espace d'adressage « function » : des registres). let = valeur liée une
    // fois pour toutes. Les types sont inférés des initialiseurs : 0.0 est un
    // f32, vec3<f32>(0.0) un vecteur de trois f32, et
    // array<f32, 19> un tableau de 19 f32.
    var streamed: array<f32, 19>;
    var density = 0.0;
    var momentum = vec3<f32>(0.0);

    // ── 1) STREAMING « PULL » + CONDITIONS AUX LIMITES ─────────────────────
    // Pour chaque direction i, on va CHERCHER la population qui arrive du
    // voisin situé en id − c_i. Trois cas :
    //   (a) le voisin sort du domaine en y  -> paroi     -> bounce-back
    //   (b) le voisin est solide            -> obstacle  -> bounce-back
    //   (c) le voisin est un nœud fluide    -> lecture normale
    // On accumule au passage la densité et la quantité de mouvement des
    // populations ARRIVÉES : ce sont précisément celles sur lesquelles la
    // collision va porter, ce qui évite une seconde passe de lecture.
    for (var i = 0u; i < 19u; i += 1u) {
        let direction = DIRECTIONS[i];
        let upstream_y = i32(id.y) - direction.y;
        var incoming: f32;
        if (upstream_y < 0 || upstream_y >= i32(dims.y)) {
            // (a) Paroi solide en y. « Halfway bounce-back » : la population
            // qui aurait dû provenir de la paroi n'existe pas, puisqu'il n'y a
            // pas de nœud fluide de l'autre côté. On la remplace par la
            // population OPPOSÉE présente ici même : elle repart d'où elle
            // venait, ce qui annule la vitesse à la paroi (condition de
            // non-glissement u = 0).
            // Le « halfway » vient du fait que la paroi théorique est placée à
            // mi-chemin entre le centre du nœud fluide et le centre du nœud
            // solide ; ce décalage d'une demi-maille est ce qui rend la
            // condition aux limites précise au 2e ordre pour un canal plan.
            // [WGSL] i32(...) est une conversion explicite u32 -> i32,
            // nécessaire car la soustraction peut devenir négative.
            incoming = populations_in[site * 19u + OPPOSITE[i]];
        } else {
            // (b)/(c) Le voisin existe en y. En x et z le domaine est
            // périodique (tunnel sans fin) : le modulo « replie » le nœud sur
            // le bord opposé, ce qui simule un domaine infini dans ces deux
            // directions.
            // [WGSL] On ajoute nx (resp. nz) AVANT le % car, comme en C, le
            // reste garde le signe du dividende : -1 % 64 vaut -1, pas 63.
            let nx = i32(dims.x);
            let nz = i32(dims.z);
            let upstream = vec3<u32>(
                u32((i32(id.x) - direction.x + nx) % nx),
                u32(upstream_y),
                u32((i32(id.z) - direction.z + nz) % nz)
            );
            if (is_solid(upstream, dims)) {
                // (b) Le voisin est solide : même rebond que sur une paroi, au
                // bord de la sphère cette fois. C'est ce mécanisme qui crée la
                // couche limite autour de l'obstacle, donc la traînée et
                // l'allée de tourbillons observées dans le sillage.
                incoming = populations_in[site * 19u + OPPOSITE[i]];
            } else {
                // (c) Cas nominal : on lit la population i du voisin amont.
                // Comme chaque invocation ne lit que ses voisins et n'écrit que
                // dans SA propre cellule, il n'y a aucun conflit d'écriture :
                // c'est tout l'intérêt du « pull » par rapport au « push »
                // (qui consisterait à écrire chez le voisin, et exigerait des
                // lectures puis des écritures atomiques).
                let upstream_site = site_index(upstream.x, upstream.y, upstream.z, dims);
                incoming = populations_in[upstream_site * 19u + i];
            }
        }
        streamed[i] = incoming;
        density += incoming;
        momentum += incoming * vec3<f32>(direction);
    }

    // ── 2) GRANDEURS MACROSCOPIQUES ────────────────────────────────────────
    // ρ = Σ f_i (accumulée ci-dessus) et u = (Σ f_i c_i) / ρ : c'est la
    // vitesse du fluide « vue » depuis ce nœud. Elle vaut 0 dans une zone au
    // repos et ≈ 0.04 (kInitialVelocity) dans l'écoulement libre à l'entrée.
    // [WGSL] La division d'un vec3<f32> par un f32 s'applique composante par
    // composante. Contrairement à la version « visualisation » du même calcul
    // (dans visualization.wgsl), aucune protection contre une densité nulle
    // n'est nécessaire : le domaine démarre à ρ = 1 et la relaxation ne l'en
    // écarte que de quelques pour cent.
    let velocity = momentum / density;

    // ── 3) COLLISION (BGK) ─────────────────────────────────────────────────
    // Chaque population est rapprochée de sa valeur d'équilibre local :
    //
    //      f_i  ←  f_i − ω · ( f_i − f_i^eq(ρ, u) )
    //
    // ω = 1/τ est le taux de relaxation (τ : temps de relaxation). Il fixe la
    // viscosité cinématique ν = (1/ω − 1/2) / 3. Ici ω = 1 (kRelaxation dans
    // src/lbm.cpp), soit ν = 1/6 en unités de réseau : un fluide peu visqueux
    // mais stable (la stabilité locale du schéma demande 0 < ω < 2).
    // On relit « streamed[i] » (déjà en registre) plutôt que populations_in :
    // on économise 19 accès mémoire par nœud à chaque pas de temps, ce qui est
    // loin d'être négligeable quand on compare à la durée du calcul lui-même.
    //
    // La valeur écrite est l'état APRÈS collision ; c'est lui qui sera
    // « streamé » au pas de temps suivant. Ce décalage d'une itération entre
    // advection et relaxation est justement ce qui rend l'algorithme
    // inconditionnellement local, et donc parallélisable sans synchronisation.
    for (var i = 0u; i < 19u; i += 1u) {
        let equilibrium_value = equilibrium(i, density, velocity);
        populations_out[site * 19u + i] = streamed[i] - params.omega * (streamed[i] - equilibrium_value);
    }
}
