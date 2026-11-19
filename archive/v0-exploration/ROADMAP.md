# Wine Classification — Roadmap

Proyecto de portafolio (data visualization / data insights) usando el dataset UCI Wine.

## Etapa 1 — Descriptive Statistics (casi completa)

- [x] Tabla descriptiva (media, std, mediana, IQR, min, max) por atributo
- [x] Histogramas de distribución por atributo
- [x] Matriz de correlación Pearson vs Spearman (heatmap dividido, significancia con Bonferroni, AlphaData)
- [x] Panel de diferencia |Pearson - Spearman| (escala fija [0,1], para verificación de independencia)
- [x] Clustering jerárquico de atributos (average linkage, distancia 1-|corr|) → cluster fenólico identificado (total_phenols, flavanoids, od_ratio, proanthocyanins)
- [x] Parallel coordinates plot coloreado por cultivar (cierre de etapa 1 / puente a etapa 2)
- [ ] (Opcional) Reordenar heatmap de correlación según orden del dendrograma
- [ ] (Opcional) Boxplots/violin por atributo, separados por cultivar

## Etapa 2 — Classification

### Rama A: LDA supervisado (reproducibilidad)
- [x] Ajustar LDA (`fitcdiscr`), con modelo completo (`wineModelFull`) y particionado (`wineModel`, Leaveout) por separado
- [x] Validar con leave-one-out CV — 98.88% obtenido, reproduce el 98.9% publicado en wine.names
- [x] Visualizar en espacio LD1 vs LD2 (`manova1` + `gscatter`), coloreado por cultivar
- [x] Guardar/usar coeficientes (`wineModelFull.Coeffs`) como insumo para etapa 3 — heatmap de pesos estandarizados (crudo × std) por par de clases + promedio absoluto, reorganizado con el cluster fenólico (Flavanoids/Total phenols) destacado con rectángulo sólido y la familia más amplia (OD280/OD315/Proanthocyanins) con rectángulo punteado

### Rama B: PCA + k-means no supervisado (descubrimiento sistemático de k)
- [x] Trabajar en espacio PCA (`pcaScore(:,1:2)`, renombrado desde `score` para no chocar con el `score` de LDA)
- [x] Determinar k sistemáticamente — loop `kRange = 2:8` con `kmeans(...,'Replicates',10)`, WCSS + silhouette promedio por k, óptimo marcado en el gráfico. Resultado: k=3 (silhouette máximo)
- [x] Ejecutar k-means con el k sugerido
- [ ] Proyectar clusters sobre plano PC1-PC2 (pendiente — falta el scatter de `pcaScore(:,1:2)` coloreado por cluster asignado, análogo al `gscatter` de LD1-LD2)
- [x] Comparar clusters descubiertos vs cultivares reales — voto mayoritario por cultivar + matriz de contingencia (`crosstab` + `heatmap`). Accuracy general: 88%. Cultivar 2 es el que más se confunde (mayormente con cultivar 1, menos con cultivar 3)
- [ ] Calcular Adjusted Rand Index en MATLAB (por ahora solo verificado del lado de Python: ARI ≈ 0.90 con 2 PCs)
- [ ] Probar hipótesis "más componentes de PCA mejoran el clustering" — comparar ARI para nPC = 2, 3, 4, 13 (ver decisión de storytelling abajo: gráfico de barras simple, no grid search nPCA×k)

### Cierre etapa 2
- [ ] Comparar desempeño LDA (supervisado) vs k-means (no supervisado)

## Etapa 3 — Twist (interpretación / insights)

- [ ] ¿Qué atributo influye más en la clasificación? (usar coeficientes de LDA)
- [ ] ¿Cómo variar atributos para que un vino se parezca a otro cultivar? (considerar que los atributos correlacionados del cluster fenólico no pueden variarse de forma independiente)

## Ideas para explorar más adelante (curiosidades, sin bloquear el avance principal)

- Comparar la correlación *dentro de cada clase* (relacionada con Sigma, la covarianza compartida de LDA) contra la correlación global de la etapa 1 — ¿se mantiene el cluster fenólico si se calcula por cultivar en vez de sobre las 178 muestras juntas?
- Para las 2 muestras mal clasificadas en el leave-one-out: revisar su margen de confianza (`scores` de `kfoldPredict`) y su posición en el scatter LD1 vs LD2, para ver si caen visualmente en la zona de traslape entre cultivares.

## Hallazgos para la etapa 3 (ya verificados, listos para usar en el "twist")

- **Cultivar 2 se parece más a cultivar 1 que a cultivar 3** — confirmado por dos vías independientes: (a) el patrón de confusión de k-means en la matriz de contingencia, y (b) la distancia de Mahalanobis entre medias de clase (usando la Σ de LDA): cultivar 1-2 = 5.34, cultivar 2-3 = 5.98, cultivar 1-3 = 7.75 (la pareja más distinta de las tres). `stats.gmdist` de `manova1` da los mismos números pero al cuadrado (28.51 / 35.80 / 60.03) — es la distancia de Mahalanobis sin la raíz cuadrada, no un cálculo distinto.
- **Más componentes de PCA no mejora necesariamente el clustering no supervisado** — probado empíricamente (ARI: 0.895 con 2 PCs, 0.880 con 3, 0.847 con 4, 0.897 con los 13). Explicación: PC3 y PC4 apenas distinguen cultivares (ver medias por PC calculadas en sesiones anteriores), así que agregarlos como dimensiones extra solo diluye la distancia euclidiana que usa k-means — a diferencia de LDA, que pondera cada dimensión por su poder discriminativo vía Σ⁻¹, k-means trata todas las dimensiones por igual.

## Decisiones de diseño ya tomadas (para no repetir la discusión)

- Se descartó SOM para el método no supervisado: no resuelve mejor el problema de "descubrir k" que k-means + análisis de silueta/codo, y el dataset (178 muestras, bien separable) no justifica sus fortalezas (pensado para datos más grandes/topológicamente complejos).
- Escalas de color en heatmaps: fijas y explícitas (no auto-escaladas), elegidas deliberadamente según el propósito de cada panel — no simplemente el default de MATLAB.
- Uso de `AlphaData` en vez de trucos de colormap con NaN, para mayor robustez/portabilidad entre versiones de MATLAB.
- Coeficientes de LDA comparados entre atributos multiplicando el coeficiente crudo por la std de cada atributo (equivalente a ajustar el modelo sobre datos estandarizados, sin necesidad de mantener un segundo modelo ni de estandarizar muestras nuevas).
- Al resumir los 3 coeficientes pairwise en una sola medida de "importancia", se usa el promedio del valor absoluto (no el promedio con signo), porque el signo puede variar entre comparaciones para un mismo atributo sin que eso signifique menor influencia.
- Convención visual para el cluster de atributos: rectángulo sólido = cluster "núcleo duro" (corte formal del dendrograma), rectángulo punteado = familia más amplia y moderadamente correlacionada (estructura anidada de dos niveles).
- Para mostrar el hallazgo de "más PCs no ayuda": NO se hará un grid search nPCA × k con heatmap de accuracy (se ve como fuerza bruta sin justificación, y diluye el mensaje de "descubrimiento sistemático, no exploración ciega"). En su lugar: k se sigue determinando sistemáticamente para cada nPC candidato, y se compara un solo gráfico de barras (ARI vs. nPC ∈ {2,3,4,13}), presentado como verificación de una hipótesis explícita, no como búsqueda exploratoria.

## Resumen conceptual — etapa 2 (para no repetir las explicaciones)

**LDA**
- Es invariante a la escala de los atributos (el coeficiente se reajusta exactamente para compensar cualquier reescalado por atributo) — por eso no hace falta estandarizar antes de `fitcdiscr`.
- Σ (covarianza compartida entre clases) es lo que hace que el método sea *Linear* en vez de *Quadratic* — mide dispersión dentro de cada clase (pooled), no la correlación global del dataset completo (son conceptos relacionados pero no idénticos).
- `.Coeffs(i,j)` son diferencias entre funciones discriminantes por clase (`wₖ = Σ⁻¹μₖ`), no clasificadores pairwise independientes — por eso nunca hay resultados cíclicos/inconsistentes entre comparaciones, a diferencia de un ensamble uno-contra-uno genuino.
- Para comparar coeficientes entre atributos de distinta escala: coeficiente crudo × std del atributo (evita mantener un segundo modelo estandarizado).
- Leave-one-out (`Leaveout`) da una estimación honesta de generalización porque cada predicción viene de un modelo que nunca vio esa muestra, sin sacrificar apenas datos de entrenamiento — no porque los 178 submodelos sean distintos entre sí (de hecho son casi idénticos, y esa estabilidad es la razón por la que el método funciona bien con pocos datos).

**k-means y validación de clusters**
- El objetivo explícito de k-means es minimizar la distancia dentro de cada cluster (WCSS) — no hay un término que maximice directamente la distancia entre centroides, aunque por la identidad `Total = Dentro + Entre` (constante para el dataset), minimizar uno maximiza el otro matemáticamente. A diferencia de LDA, no hay normalización por dispersión (LDA optimiza una razón; k-means, sumas crudas).
- Silhouette score por punto: compara la distancia promedio dentro del propio cluster contra la distancia promedio a *todos* los puntos del cluster vecino más cercano (no al centroide, no al punto más próximo).
- k-means es no determinista (centroides iniciales aleatorios) — mitigado con `'Replicates'` (varias inicializaciones, se queda con la mejor).
- Evaluar un clustering no supervisado contra etiquetas reales requiere resolver el problema de que las etiquetas de cluster son arbitrarias (permutación) — voto mayoritario por clase es un enfoque válido pero puede esconder que dos clases reales caigan en el mismo cluster mayoritario; Adjusted Rand Index lo maneja de forma más rigurosa y simétrica.
- La distancia de Mahalanobis entre medias de clase (misma Σ de LDA) es una forma directa de medir "qué tan parecidas son dos clases", independiente de cualquier corrida específica de clustering.
