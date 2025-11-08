// =====================================================================
// STEP 1: Earth Engine Export Script
// Exports training samples (AlphaEarth + BRBC labels) to Cloud Storage
// =====================================================================

// -----------------------------
// Configuration
// -----------------------------
var RANDOM_SEED = 42;
var SCALE = 50;
var PROJECT_ID = 'wetmaps-476922';
var BUCKET_NAME = 'wetmaps';

var canonicalClassDict = ee.Dictionary({
  'Marsh': 0,
  'Shallow Open Water': 1,
  'Swamp': 2,
  'Fen (Graminoid)': 3,
  'Fen (Woody)': 3
});

print('=== EARTH ENGINE EXPORT FOR CLOUD RUN ===');

// -----------------------------
// Study Area (Calgary - Test Region)
// -----------------------------
var calgaryBounds = ee.Geometry.Rectangle([-114.3, 50.8, -113.7, 51.3]);
Map.centerObject(calgaryBounds, 11);

// -----------------------------
// Load Ground Truth (BRBC)
// -----------------------------
var brbc = ee.FeatureCollection('projects/wetmaps-476922/assets/WetlandInventory_CoverTypeInfo');

var brbcFiltered = brbc
  .filter(ee.Filter.inList('WetlandCla', canonicalClassDict.keys()))
  .filterBounds(calgaryBounds)
  .map(function(f) {
    return f.simplify({maxError: 50});
  });

// Encode class labels
var brbcEncoded = brbcFiltered.map(function(f) {
  var classIndex = canonicalClassDict.get(f.get('WetlandCla'));
  return f.set('classIndex', classIndex);
});

print('Ground truth features:', brbcEncoded.size());

// -----------------------------
// Load AlphaEarth (Calgary)
// -----------------------------
var alphaEarth = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterBounds(calgaryBounds)
  .filterDate('2024-01-01', '2024-12-31');

var alphaEarthExists = alphaEarth.size().gt(0);
var alphaEarth2024 = ee.Image(ee.Algorithms.If(
  alphaEarthExists,
  alphaEarth.first().clip(calgaryBounds),
  ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(calgaryBounds)
    .filterDate('2022-05-01', '2022-09-30')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
    .median()
    .clip(calgaryBounds)
));

print('AlphaEarth bands:', alphaEarth2024.bandNames().size());

// -----------------------------
// Balanced Sampling (500 per class)
// -----------------------------
print('');
print('=== SAMPLING TRAINING DATA ===');

var samplesPerClass = 714; // Will split 70/15/15 = 500/107/107

var class0 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 0)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: true, // Keep geometries for GeoTIFF export
  tileScale: 16
}).randomColumn('random', RANDOM_SEED).limit(samplesPerClass);

var class1 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 1)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: true,
  tileScale: 16
}).randomColumn('random', RANDOM_SEED).limit(samplesPerClass);

var class2 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 2)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: true,
  tileScale: 16
}).randomColumn('random', RANDOM_SEED).limit(samplesPerClass);

var class3 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 3)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: true,
  tileScale: 16
}).randomColumn('random', RANDOM_SEED).limit(samplesPerClass);

var allSamples = class0.merge(class1).merge(class2).merge(class3);

print('Total samples:', allSamples.size());
print('Class distribution:', allSamples.aggregate_histogram('classIndex'));

// -----------------------------
// Export Training Samples to Cloud Storage
// -----------------------------
print('');
print('=== EXPORTING TO CLOUD STORAGE ===');

Export.table.toCloudStorage({
  collection: allSamples,
  description: 'wetland_training_samples_calgary',
  bucket: BUCKET_NAME,
  fileNamePrefix: 'training_data/calgary_samples',
  fileFormat: 'CSV',
  selectors: ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 
              'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
              'classIndex', '.geo'] // .geo includes geometry
});

// -----------------------------
// Export Full AlphaEarth Image for Inference
// -----------------------------
Export.image.toCloudStorage({
  image: alphaEarth2024,
  description: 'alphaearth_calgary_full',
  bucket: BUCKET_NAME,
  fileNamePrefix: 'inference_data/calgary_alphaearth',
  region: calgaryBounds,
  scale: SCALE,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

print('');
print('=== EXPORT TASKS CREATED ===');
print('✓ Training samples CSV');
print('✓ Full AlphaEarth GeoTIFF for inference');
print('');
print('NEXT STEPS:');
print('1. Click "Tasks" tab and run both exports');
print('2. Wait for exports to complete (check Cloud Storage)');
print('3. Run Cloud Run deployment script');
print('4. Trigger training via Colab notebook');

// Visualize sampling locations
Map.addLayer(brbcEncoded, {color: 'yellow'}, 'Ground Truth Polygons');
Map.addLayer(allSamples, {color: 'red'}, 'Training Sample Points');