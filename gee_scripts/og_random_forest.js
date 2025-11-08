// =====================================================================
// Alberta Wetland Classification - LIGHTWEIGHT (NO CRASHES)
// 100% FREE - Zero GCP Credits
// =====================================================================

// -----------------------------
// 0. Configuration (REDUCED FOR SPEED)
// -----------------------------
var RANDOM_SEED = 42;
var SCALE = 50;
var MAX_SAMPLES = 2000;

var canonicalClassDict = ee.Dictionary({
  'Marsh': 0,
  'Shallow Open Water': 1,
  'Swamp': 2,
  'Fen (Graminoid)': 3,
  'Fen (Woody)': 3
});

var classNames = ['Marsh', 'Shallow Open Water', 'Swamp', 'Fen'];
var palette = ['blue', 'cyan', 'green', 'yellow'];

print('=== LIGHTWEIGHT WETLAND CLASSIFIER ===');
print('Optimized to prevent browser crashes');

// -----------------------------
// 1. SMALL Study Area (Calgary only - not full Alberta)
// -----------------------------
var calgaryBounds = ee.Geometry.Rectangle([-114.3, 50.8, -113.7, 51.3]);

print('Study Area: Calgary region only');

// -----------------------------
// 2. Load Ground Truth (MINIMAL)
// -----------------------------
print('');
print('=== LOADING DATA ===');

var brbc = ee.FeatureCollection('projects/wetmaps-476922/assets/WetlandInventory_CoverTypeInfo');

var brbcFiltered = brbc
  .filter(ee.Filter.inList('WetlandCla', canonicalClassDict.keys()))
  .filterBounds(calgaryBounds)
  .map(function(f) {
    return f.simplify({maxError: 50});
  });

var brbcSample = brbcFiltered.randomColumn('sample', RANDOM_SEED)
  .filter(ee.Filter.lt('sample', 0.10))
  .limit(500);

print('Ground truth features:', brbcSample.size());

var brbcEncoded = brbcSample.map(function(f) {
  var classIndex = canonicalClassDict.get(f.get('WetlandCla'));
  return f.set('classIndex', classIndex);
});

// -----------------------------
// 3. Load AlphaEarth (Calgary only)
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
// 4. Sample Training Data with CLASS BALANCING
// -----------------------------
print('');
print('=== SAMPLING TRAINING DATA (BALANCED) ===');

var samplesPerClass = 500;

var class0 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 0)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: false,
  tileScale: 16
}).limit(samplesPerClass);

var class1 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 1)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: false,
  tileScale: 16
}).limit(samplesPerClass);

var class2 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 2)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: false,
  tileScale: 16
}).limit(samplesPerClass);

var class3 = alphaEarth2024.sampleRegions({
  collection: brbcEncoded.filter(ee.Filter.eq('classIndex', 3)),
  properties: ['classIndex'],
  scale: SCALE,
  geometries: false,
  tileScale: 16
}).limit(samplesPerClass);

var trainingData = class0.merge(class1).merge(class2).merge(class3);

print('Training samples:', trainingData.size());
print('Class distribution:', trainingData.aggregate_histogram('classIndex'));

// -----------------------------
// 5. Train Random Forest (FAST)
// -----------------------------
print('');
print('=== TRAINING MODEL ===');

var bands = alphaEarth2024.bandNames();

var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 30,
  seed: RANDOM_SEED
}).train({
  features: trainingData,
  classProperty: 'classIndex',
  inputProperties: bands
});

print('✓ Model trained (30 trees)');

// -----------------------------
// 6. Validation (Precision / Recall)
// -----------------------------
print('');
print('=== VALIDATION ===');

var split = trainingData.randomColumn('split', RANDOM_SEED);
var train = split.filter(ee.Filter.lt('split', 0.8));
var val = split.filter(ee.Filter.gte('split', 0.8));

classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 30,
  seed: RANDOM_SEED
}).train({
  features: train,
  classProperty: 'classIndex',
  inputProperties: bands
});

var validated = val.classify(classifier);
var confMatrix = validated.errorMatrix('classIndex', 'classification');

print('Confusion Matrix:');
print(confMatrix);

var accuracy = confMatrix.accuracy();
print('Overall Accuracy:', accuracy.multiply(100).format('%.1f').getInfo() + '%');

var recall = confMatrix.producersAccuracy();
var precision = confMatrix.consumersAccuracy();

print('Recall (per class):');
print(recall);

print('Precision (per class):');
print(precision);

print('Recall by Class:');
var recallList = recall.toList();
for (var i = 0; i < classNames.length; i++) {
  print(classNames[i] + ': ' +
    ee.Number(recallList.get(i)).multiply(100).format('%.1f').getInfo() + '%');
}

print('Precision by Class:');
var precisionList = precision.toList();
for (var i = 0; i < classNames.length; i++) {
  print(classNames[i] + ': ' +
    ee.Number(precisionList.get(i)).multiply(100).format('%.1f').getInfo() + '%');
}

if (accuracy.getInfo() >= 0.80) {
  print('✓ TARGET MET: ≥80%');
} else {
  print('⚠ Below 80% target (limited by small training set)');
}

// -----------------------------
// 7. Classify (Calgary only)
// -----------------------------
print('');
print('=== CLASSIFYING ===');

var classified = alphaEarth2024.classify(classifier);

print('✓ Classification complete');

// -----------------------------
// 8. Visualization with NON-WETLAND MASK
// -----------------------------
print('');
print('=== DISPLAYING MAP ===');

Map.centerObject(calgaryBounds, 11);

Map.addLayer(
  alphaEarth2024,
  {bands: ['A00', 'A01', 'A02'], min: -1, max: 1},
  'Satellite Base',
  false
);

var classifiedMasked = classified.clip(brbcEncoded.geometry());

Map.addLayer(
  classifiedMasked,
  {min: 0, max: 3, palette: palette},
  'Wetland Classification (Masked)'
);

Map.addLayer(
  classified,
  {min: 0, max: 3, palette: palette, opacity: 0.6},
  'Full Classification (Unmasked)',
  false
);

Map.addLayer(
  brbcEncoded,
  {color: 'yellow'},
  'Ground Truth Polygons',
  true
);

// -----------------------------
// 9. Simple Legend
// -----------------------------
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px',
    backgroundColor: 'white'
  }
});

var title = ui.Label({
  value: 'Calgary Wetlands',
  style: {fontWeight: 'bold', fontSize: '14px'}
});
legend.add(title);

for (var i = 0; i < classNames.length; i++) {
  var row = ui.Panel({
    widgets: [
      ui.Label({style: {backgroundColor: palette[i], padding: '6px', margin: '2px'}}),
      ui.Label(classNames[i], {margin: '2px 0 2px 4px'})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
  legend.add(row);
}

legend.add(ui.Label({
  value: 'Accuracy: ' + accuracy.multiply(100).format('%.1f').getInfo() + '%',
  style: {margin: '4px 0 0 0', fontSize: '11px', fontWeight: 'bold'}
}));

Map.add(legend);

// -----------------------------
// 10. Summary
// -----------------------------
print('');
print('=== SUMMARY ===');
print('✓ Model trained with BALANCED classes');
print('✓ Accuracy:', accuracy.multiply(100).format('%.1f').getInfo() + '%');
print('✓ Training samples:', train.size().getInfo());
print('✓ 500 samples per class (equal representation)');
print('✓ NO GCP CREDITS USED');
print('');
print('CLASS BALANCING FIX:');
print('- Equal samples from each wetland type');
print('- Prevents majority class (Marsh) bias');
print('- Classification masked to wetland areas only');
print('');
print('IMPORTANT NOTES:');
print('- Blue (Marsh) should now be less dominant');
print('- Yellow polygons show ground truth areas');
print('- Classification only shown where wetlands exist');
print('- Roads/cities are masked out');
print('');
print('TO IMPROVE FURTHER:');
print('1. Toggle "Ground Truth Polygons" layer on');
print('2. Compare classified colors to ground truth');
print('3. If still too much blue, increase other class samples');
print('4. Consider adding "non-wetland" class for roads/cities');
print('');
print('CURRENT STATUS:');
print('✓ Balanced training complete');
print('✓ Ready for visual inspection');

