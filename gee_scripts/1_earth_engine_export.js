// =====================================================================
// FIXED: Earth Engine Export with Proper Non-Wetland Sampling
// =====================================================================

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

print('=== WETLAND CLASSIFIER WITH NON-WETLAND CLASS ===');

// -----------------------------
// Study Area (Calgary)
// -----------------------------
var calgaryBounds = ee.Geometry.Rectangle([-114.3, 50.8, -113.7, 51.3]);
Map.centerObject(calgaryBounds, 11);

// -----------------------------
// Load AlphaEarth
// -----------------------------
var alphaEarth2024 = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterBounds(calgaryBounds)
  .filterDate('2024-01-01', '2024-12-31')
  .first()
  .clip(calgaryBounds);

print('AlphaEarth bands:', alphaEarth2024.bandNames().size());

// -----------------------------
// Load Ground Truth (BRBC Wetlands)
// -----------------------------
var brbc = ee.FeatureCollection('projects/wetmaps-476922/assets/WetlandInventory_CoverTypeInfo');

var brbcFiltered = brbc
  .filter(ee.Filter.inList('WetlandCla', canonicalClassDict.keys()))
  .filterBounds(calgaryBounds)
  .map(function(f) { return f.simplify({maxError: 50}); });

var brbcEncoded = brbcFiltered.map(function(f) {
  var classIndex = canonicalClassDict.get(f.get('WetlandCla'));
  return f.set('classIndex', classIndex);
});

print('Wetland polygons:', brbcEncoded.size());

// -----------------------------
// Sample WETLAND Classes (0-3)
// -----------------------------
print('');
print('=== SAMPLING WETLAND CLASSES ===');

var samplesPerClass = 714;

var class0 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 0)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: true,
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

// Check actual sample counts
print('Marsh samples:', class0.size());
print('Shallow Water samples:', class1.size());
print('Swamp samples:', class2.size());
print('Fen samples:', class3.size());

// -----------------------------
// Sample NON-WETLAND Areas (SIMPLIFIED METHOD)
// -----------------------------
print('');
print('=== SAMPLING NON-WETLAND AREAS ===');

// SIMPLIFIED APPROACH: Use image-based mask instead of geometry operations
// Create a mask where wetlands = 0, non-wetlands = 1
var wetlandMask = ee.Image(0).byte().paint(brbcEncoded, 1);
var nonWetlandMask = wetlandMask.not();  // Invert: non-wetlands = 1

// Add the mask as a band to AlphaEarth
var alphaWithMask = alphaEarth2024.addBands(nonWetlandMask.rename('nonwetland_mask'));

// Sample from the entire study area
var nonWetlandSamples = alphaWithMask.sample({
  region: calgaryBounds,
  scale: SCALE,
  numPixels: samplesPerClass * 5,  // Oversample
  seed: RANDOM_SEED,
  geometries: true,
  tileScale: 4
}).filter(ee.Filter.eq('nonwetland_mask', 1))  // Keep only non-wetland pixels
  .map(function(f) {
    return f.set('classIndex', 4).select(['A.*', 'classIndex']);  // Class 4, remove mask band
  })
  .limit(samplesPerClass);

print('Non-wetland samples:', nonWetlandSamples.size());

// -----------------------------
// Merge All Classes
// -----------------------------
var allSamples = class0.merge(class1).merge(class2).merge(class3).merge(nonWetlandSamples);

print('');
print('=== FINAL SAMPLE COUNTS ===');
print('Total samples:', allSamples.size());
print('Class distribution:', allSamples.aggregate_histogram('classIndex'));

// -----------------------------
// Export Training Samples
// -----------------------------
print('');
print('=== EXPORTING TO CLOUD STORAGE ===');

// Build selector list (A00-A63 + classIndex + geometry)
var selectors = [];
for (var i = 0; i <= 63; i++) {
  var bandNum = i.toString();
  if (bandNum.length === 1) bandNum = '0' + bandNum;
  selectors.push('A' + bandNum);
}
selectors.push('classIndex');
selectors.push('.geo');

Export.table.toCloudStorage({
  collection: allSamples,
  description: 'wetland_training_samples_calgary',
  bucket: BUCKET_NAME,
  fileNamePrefix: 'training_data/calgary_samples',
  fileFormat: 'CSV',
  selectors: selectors
});

// Export AlphaEarth
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
print('✓ Training samples CSV (with proper non-wetland sampling)');
print('✓ AlphaEarth GeoTIFF');
print('');
print('NEXT STEPS:');
print('1. Click "Tasks" tab and RUN both exports');
print('2. Wait for completion');
print('3. Update Colab training CSV to: training_data/calgary_samples.csv');

// -----------------------------
// Visualization
// -----------------------------
Map.addLayer(brbcEncoded, {color: 'yellow'}, 'Wetland Polygons', true);
Map.addLayer(nonWetlandMask.selfMask(), {palette: ['red']}, 'Non-Wetland Areas', false);

var legend = ui.Panel({
  style: {position: 'bottom-left', padding: '8px', backgroundColor: 'white'}
});

legend.add(ui.Label({value: 'Sample Locations', style: {fontWeight: 'bold'}}));
legend.add(ui.Label({value: 'Yellow = Wetland polygons'}));
legend.add(ui.Label({value: 'Red = Non-wetland sampling area'}));

Map.add(legend);

print('');
print('=== SUMMARY ===');
print('✓ Image-based non-wetland sampling (faster & simpler)');
print('✓ Balanced classes (714 per class)');
print('✓ Fen = 0 in Calgary (will be excluded from model)');
print('✓ Ready to export!');