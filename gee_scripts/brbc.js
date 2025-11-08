// -----------------------------
// 0. Setup
// -----------------------------
var calgaryBounds = ee.Geometry.Rectangle([-114.3, 50.8, -113.7, 51.3]);
var brbc = ee.FeatureCollection('projects/wetmaps-476922/assets/WetlandInventory_CoverTypeInfo');

var canonicalClassDict = ee.Dictionary({
  'Marsh': 0,
  'Shallow Open Water': 1,
  'Swamp': 2,
  'Fen (Graminoid)': 3,
  'Fen (Woody)': 3
});

// Filter BRBC to only the classes we care about
var brbcFiltered = brbc.filter(ee.Filter.inList('WetlandCla', canonicalClassDict.keys()));

// Log unique classifications
var uniqueClasses = brbcFiltered.aggregate_array('WetlandCla').distinct();
print('Unique classifications in BRBC:', uniqueClasses);

// -----------------------------
// 1. Filter Calgary & valid wetland classes
// -----------------------------
var wetlandClasses = ['Marsh', 'Shallow Open Water', 'Swamp', 'Fen (Graminoid)', 'Fen (Woody)'];
var palette = ['blue', 'cyan', 'green', 'yellow', 'brown'];

var brbcFiltered = brbc.filter(ee.Filter.inList('WetlandCla', wetlandClasses));
var brbcCalgary = brbcFiltered.filterBounds(calgaryBounds);

// -----------------------------
// 2. Encode classes safely
// -----------------------------
var classDict = ee.Dictionary({
  'Marsh': 0,
  'Shallow Open Water': 1,
  'Swamp': 2,
  'Fen (Graminoid)': 3,
  'Fen (Woody)': 4
});

var brbcEncoded = brbcCalgary.map(function(f) {
  var classIndex = classDict.get(f.get('WetlandCla'));
  classIndex = ee.Algorithms.If(classIndex, classIndex, -1);
  return f.set('classIndex', classIndex);
});

// -----------------------------
// 3. Rasterize using paint (better for small polygons)
// -----------------------------
var empty = ee.Image().byte();
var brbcRaster = empty.paint({
  featureCollection: brbcEncoded,
  color: 'classIndex'
}).clip(calgaryBounds);

brbcRaster = brbcRaster.rename('classIndex');

// -----------------------------
// 4. Display
// -----------------------------
Map.centerObject(calgaryBounds, 11);
Map.addLayer(
  brbcRaster.updateMask(brbcRaster.gte(0)),
  {min:0, max:4, palette: palette},
  'Calgary Wetlands'
);

// -----------------------------
// 5. Summary
// -----------------------------
print('Number of Calgary features:', brbcCalgary.size());
print('Unique wetland classes in Calgary:', brbcCalgary.aggregate_array('WetlandCla').distinct());

// -----------------------------
// Legend
// -----------------------------
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

var legendTitle = ui.Label({
  value: 'Calgary Wetland Types',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 4px 0', padding: '0'}
});
legend.add(legendTitle);

var names = ['Marsh', 'Shallow Open Water', 'Swamp', 'Fen (Graminoid)', 'Fen (Woody)'];
var colors = ['blue', 'cyan', 'green', 'yellow', 'brown'];

for (var i = 0; i < names.length; i++) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: colors[i],
      padding: '8px',
      margin: '0 0 4px 0'
    }
  });
  
  var description = ui.Label({
    value: names[i],
    style: {margin: '0 0 4px 6px'}
  });
  
  var row = ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
  
  legend.add(row);
}

Map.add(legend);

