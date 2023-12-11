## Uitleg

In deze folder staat alles voor het trainen van een beeldherkenningsmodel voor nachtvlinders. In de notebooks is uitleg bij de code te vinden.

### Achtergrond informatie

- Hoe werkt beeldherkenning? [Convolutional Neural Networks (CNNs) explained](https://www.youtube.com/watch?v=YRhxdVk_sIs)
- Hoe goed presteert het model? [Evalueren van multi-class problems met accuracy, recall en precision](https://www.evidentlyai.com/classification-metrics/multi-class-metrics).

## Notebooks

- **Beeldherkenning.ipynb**: Train de beeldherkenning voor nachtvlinders (of iets anders).
- **DataVoorbereiden.ipynb**: Combineer verschillende datasets tot één gestandaardiseerde, gefilterde dataset.
- **Scrape<...>Photos.ipynb**: Scripts voor het scrapen van respectievelijk GBIF, Lepiforum (LFP) en het DVS Meetnet.


## Folders

- **dataset**: Opgeschoonde dataset. Resultaat van `DataVoorbereiden.ipynb`.
- **runs**: Tensorboard data. Als de server draait, is de data [hier](http://http://localhost:6006) te bekijken.

## Overige bestanden

- **20231109_134031_19_idx_to_class.csv**: Mapping van model output indices naar nachtvlindersoortnaam (voor `20231109_134031_19_model`). Gecreëerd in `Beeldherkenning.ipynb`.
- **20231109_134031_19_model**: Het beste nachtvlinderherkenningsmodel van het jaar 2023.
- **20231109_134031_19_test_results_model.csv**: Testresultaten voor `20231109_134031_19_model`.
- **mothUtils.py**: Extra code die benodigd is voor `Beeldherkenning.ipynb`.
- **yolov5s.pt**: Weights voor het getrainde YOLO model (uit `/data/mothDetection/...`).
