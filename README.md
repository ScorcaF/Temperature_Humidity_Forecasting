# Temperature_Humidity_Forecasting
Multi-output models for temperature and humidity forecasting on Raspberry 4. The objective is to deploy light models able to infer multi-step predictions on the Jena
Climate dataset: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip

![th_forecasting](https://user-images.githubusercontent.com/70110839/209452209-67dd8de5-bc39-4ffb-86b4-81148b8afc71.png)

The architectures trained are two-layers MLPs and
CNNs, exploiting ”Early Stopping”, namely training until the MAEs of the validation set reaches some threshold
values, for a minimum of 16 epochs. The resources optimizations
considered are Weight quantization, Structured pruning, and Magnitude-based pruning.

The results are summarized in the following table.

![th_results](https://user-images.githubusercontent.com/70110839/209452523-d992cbf8-b39a-4a21-97ad-f29c57a28e99.png)
