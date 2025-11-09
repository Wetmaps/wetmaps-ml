// =====================================================================
// Visualize Cloud Run Classification Results (WITH NON-WETLAND CLASS)
// =====================================================================

// Load classified image (UPDATE YOUR_USERNAME!)
// Load classified image
var classified = ee.Image('projects/wetmaps-476922/assets/calgary_classified');

// Mask out any pixels that are class 3 (Fen) - these are artifacts
var classifiedClean = classified.updateMask(classified.neq(3));

// Define visualization (5 classes now!)
var palette = ['#a6d96a', 'blue', 'green', 'yellow', '#d3d3d3'];
var classNames = ['Marsh', 'Shallow Open Water', 'Swamp', 'Fen', 'Non-Wetland'];

// Center on Calgary
Map.setCenter(-114.0, 51.05, 11);

// Add classification layer
Map.addLayer(
  classifiedClean,
  {min: 0, max: 4, palette: palette},  // max: 4 for 5 classes
  'Wetland Classification (Cloud Run)'
);

// Create legend
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px',
    backgroundColor: 'white'
  }
});

var title = ui.Label({
  value: 'Calgary Wetlands + Non-Wetland',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 8px 0'}
});
legend.add(title);

// Add color boxes
for (var i = 0; i < classNames.length; i++) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: palette[i],
      padding: '8px',
      margin: '2px 8px 2px 0'
    }
  });
  
  var label = ui.Label({
    value: classNames[i],
    style: {margin: '2px 0'}
  });
  
  var row = ui.Panel({
    widgets: [colorBox, label],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
  
  legend.add(row);
}

// Add info
legend.add(ui.Label({
  value: 'Cloud Run Random Forest (5 classes)',
  style: {margin: '8px 0 0 0', fontSize: '11px', fontStyle: 'italic'}
}));

legend.add(ui.Label({
  value: 'Gray = Cities/Roads/Agriculture',
  style: {margin: '2px 0 0 0', fontSize: '10px', color: '#666'}
}));

Map.add(legend);

print('✓ Classification loaded');
print('✓ 5-class wetland map (includes non-wetland)');
print('✓ Cities and roads should now show as gray');