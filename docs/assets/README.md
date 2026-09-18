# Figure provenance

`event_time_target.svg` and its PNG preview are an authored conceptual diagram. They show 64 illustrative historical event locations with varying gaps and five illustrative prices in the inclusive future label interval. The source is the public example's task definition, not a market dataset or an experiment result.

The horizontal origin is the prediction endpoint `t_i`; the model's own time input is elapsed time from the first event of the historical window. The illustrated historical span is arbitrary. Only the future 2.5–3.5 second bounds and 64-event count correspond to example configuration values.

Blue marks denote observed history; orange marks denote future prices used in the label. Gray marks outside the future band are not included in that label. The illustration does not imply that the model receives future prices.

The SVG contains native editable vector/text elements. The PNG is included for viewers that do not render SVG. Neither image demonstrates predictive performance.
