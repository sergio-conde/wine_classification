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
- [ ] Trabajar en espacio PCA (PC1+PC2 ya validado: separan bien con ~55% de varianza)
- [ ] Determinar k sistemáticamente (método del codo / silhouette score vs k) — sin asumir 3 de antemano
- [ ] Ejecutar k-means con el k sugerido
- [ ] Proyectar clusters sobre plano PC1-PC2
- [ ] Comparar clusters descubiertos vs cultivares reales (matriz de contingencia / Adjusted Rand Index)

### Cierre etapa 2
- [ ] Comparar desempeño LDA (supervisado) vs k-means (no supervisado)

## Etapa 3 — Twist (interpretación / insights)

- [ ] ¿Qué atributo influye más en la clasificación? (usar coeficientes de LDA)
- [ ] ¿Cómo variar atributos para que un vino se parezca a otro cultivar? (considerar que los atributos correlacionados del cluster fenólico no pueden variarse de forma independiente)

## Ideas para explorar más adelante (curiosidades, sin bloquear el avance principal)

- Comparar la correlación *dentro de cada clase* (relacionada con Sigma, la covarianza compartida de LDA) contra la correlación global de la etapa 1 — ¿se mantiene el cluster fenólico si se calcula por cultivar en vez de sobre las 178 muestras juntas?
- Para las 2 muestras mal clasificadas en el leave-one-out: revisar su margen de confianza (`scores` de `kfoldPredict`) y su posición en el scatter LD1 vs LD2, para ver si caen visualmente en la zona de traslape entre cultivares.

## Decisiones de diseño ya tomadas (para no repetir la discusión)

- Se descartó SOM para el método no supervisado: no resuelve mejor el problema de "descubrir k" que k-means + análisis de silueta/codo, y el dataset (178 muestras, bien separable) no justifica sus fortalezas (pensado para datos más grandes/topológicamente complejos).
- Escalas de color en heatmaps: fijas y explícitas (no auto-escaladas), elegidas deliberadamente según el propósito de cada panel — no simplemente el default de MATLAB.
- Uso de `AlphaData` en vez de trucos de colormap con NaN, para mayor robustez/portabilidad entre versiones de MATLAB.
- Coeficientes de LDA comparados entre atributos multiplicando el coeficiente crudo por la std de cada atributo (equivalente a ajustar el modelo sobre datos estandarizados, sin necesidad de mantener un segundo modelo ni de estandarizar muestras nuevas).
- Al resumir los 3 coeficientes pairwise en una sola medida de "importancia", se usa el promedio del valor absoluto (no el promedio con signo), porque el signo puede variar entre comparaciones para un mismo atributo sin que eso signifique menor influencia.
- Convención visual para el cluster de atributos: rectángulo sólido = cluster "núcleo duro" (corte formal del dendrograma), rectángulo punteado = familia más amplia y moderadamente correlacionada (estructura anidada de dos niveles).
