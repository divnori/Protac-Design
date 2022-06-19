
# Random Forest Surrogate Model


The random forest surrogate model was trained on the PROTAC-DB dataset, a web-based open-access database that integrates structural information and experimental data of PROTACs. The database contains information from several literature sources to display the efficacy of specific PROTAC tertiary complexes in given cell types. The tertiary complex includes a target protein, an E3 ligase, and PROTAC molecule. 

The following steps were taken to train a random forest model on this data. This model will be used to develop a scoring function to evaluate potential protein degradation of a proposed PROTAC molecule. This scoring function will be integrated into a GraphINVENT molecular generation neural network.

### Pre-Processing Data

1. Read in protac.csv which includes a protac SMILES string, E3 ligase ID label, and target protein ID label
2. Create 7 classes of E3 ligases - CRBN, VHL, IAP, MDM2, DCAF, AhR, RNF - by grouping some entries with matching prefixes/suffixes together
3. Drop data points with no DC50 information. DC50 is defined as the concentration where 50% of the target protein has been degraded, and this value will be processed to be the response variable
4. One hot encode the target and E3 ligase features. At this point, the data set contained 638 records with 89 features
5. Get the Morgan molecular fingerprint for each PROTAC as a 256 bit array based on the SMILES string. Add this array to the dataset to create 256 new features. This was done using the rdkit library
6. Add a cell type feature by looking at the experimental notes from protac.csv. Use regex to process this information. 109 cell types were detected

## Model Training - Binary Response Variable

1. Create a new response variable column
2. If the word "degradation" is observed in a record, label it with a 1 in the response column. If not, label it with a 0
3. Create train and test model matrices and use a 80/20 train-test split
4. Train a binary random forest classifier and record metrics across three min_samples_leaf values

### Best Model Metrics

#### AUC of 90.53% in train & 80.23% in test

## Model Training - DC50 Response Variable

1. Process the DC50 column by removing special characters, letters, etc. but leaving the "/" character
2. In rows with multiple DC50 values separated by "/", split out these values by cell type. For example, a record containing "HLF/SNU-398/HUCCT1" in the cell type column and "40.0/15.6/35.7" in the DC50 column should be split into three rows: "HLF | 40.0","SNU-398 | 15.6", "HUCCT1 | 35.7"
3. Drop all rows with an outlier DC50 value to improve model accuracy. This includes values greater than 100 and less than or equal to 0
4. Plot the distribution of DC50 values and determine three cut-off thresholds to break DC50 values into four buckets. These thresholds were determined to be 4.0, 15.6, and 50.0
5. If a row's DC50 value is less than or equal to 4.0, place a 0 in the response variable column. Subsequently, if a row's DC50 value is between 4.0 and 15.6, place a 1 in the response variable column. Continue until 4 DC50 buckets are created
6. Train a multiclass classifier random forest model and record metrics

### Best Model Metrics

#### Currently undetermined
