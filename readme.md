# Amazon Review Detector

A machine learning system to detect computer-generated Amazon reviews using deep learning and Kafka streaming.

## Architecture

The project follows a two-phase architecture:

### Phase 1: Data Collection and Processing using Amazon Scraper

- **Data Collection**: Collects review data using Amazon Scraper API
- **Data Preprocessing**: Cleans and transforms data into feature vectors
- **Feature Extraction**: Extracts TF-IDF and linguistic features
- **Clustering and Model Development**: Develops multiple models (NN, XGBoost, Random Forest)
- **Evaluation**: Comprehensive evaluation of model performance

### Phase 2: Real-time Data Processing using Apache Kafka

- **Data Sources**: Multiple sources including Amazon API and external data
- **Stream Processing**: Uses Kafka for real-time data streaming
- **Real-time Prediction**: Applies trained models to new data streams
- **Preprocessing & Feature Extraction**: Real-time feature extraction pipeline
- **Prediction and Evaluation**: Live evaluation of review authenticity

## Features

### Enhanced Data Processing

- Advanced TF-IDF vectorization:
  - N-gram ranges (1-3 words)
  - Intelligent feature selection
  - Rare and common word filtering
- Linguistic feature extraction:
  - Word and sentence statistics
  - Vocabulary richness analysis
  - Writing style metrics
- Data augmentation capabilities
- Robust train/validation/test splitting

### Improved Model Architecture

- Deep neural network with:
  - Multiple dense layers (512 → 256 → 128 → 64)
  - Batch normalization
  - Graduated dropout rates
  - Advanced activation functions
- Early stopping and model checkpointing
- Class weight balancing
- Cross-validation support
- Model ensemble capabilities

### Real-time Processing

- Kafka streaming integration
- Live scoring of reviews
- Stream-based model updates
- Real-time monitoring dashboard
- Configurable processing pipeline

## Usage

1. **Environment Setup**:

   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Start Kafka and related services
   docker-compose up -d
   ```

2. **Data Collection and Training (Phase 1)**:

   ```bash
   # Run the Amazon scraper to collect reviews
   python main.py --mode train --run-scraper

   # Or train with existing dataset
   python main.py --mode train
   ```

3. **Model Testing**:

   ```bash
   # Test the model on reserved data
   python main.py --mode test

   # Or test with new Amazon scraper data
   python main.py --mode test --run-scraper
   ```

4. **Real-time Analysis (Phase 2)**:

   ```bash
   # Start the Kafka consumer for real-time processing
   python main.py --mode serve

   # In another terminal, simulate a stream of reviews
   python -m src.kafka.producer --simulate 10 5
   ```

5. **Full Kafka Pipeline with Amazon Scraper**:

   ```bash
   # Run the scraper, collect data, and process it in real-time
   python main.py --mode kafka-pipeline

   # Alternative: manually send scraped data to Kafka
   python amazon_scraper.py --send-to-kafka
   ```

## Configuration

### Model Settings (`config/config.yaml`):

```yaml
kafka:
  bootstrap_servers: "localhost:39092"
  topic: "amazon_reviews"
  group_id: "review_detector"

model:
  input_features: 5000
  hidden_layers: [512, 256, 128, 64]
  dropout_rates: [0.2, 0.2, 0.2, 0.1]
  learning_rate: 0.0005
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
  use_batch_normalization: true
  use_class_weights: true
```

## Project Structure

```
├── Data/                      # Data storage
│   ├── Testing_data/          # Test datasets from training split
│   ├── Scraped_data/          # Amazon scraper data
│   │   └── Json_file/         # Raw JSON data from scraper
│   └── Training_data/         # Training datasets
├── config/                    # Configuration files
│   └── config.yaml            # Main configuration
├── models/                    # Saved models
├── results/                   # Analysis results
│   ├── training/              # Training results and visualizations
│   ├── testing/               # Testing results and metrics
│   └── real_time_predictions.csv  # Kafka real-time analysis results
├── src/                       # Source code
│   ├── kafka/                 # Kafka integration
│   │   ├── consumer.py        # Kafka consumer
│   │   └── producer.py        # Kafka producer
│   ├── processing/            # Data processing
│   │   └── data_preprocessing.py  # Data preprocessing
│   └── utils/                 # Utilities
│       └── logger.py          # Logging configuration
├── amazon_scraper.py          # Amazon review scraper
├── combine_datasets.py        # Dataset combination tool
├── docker-compose.yml         # Docker configuration
├── main.py                    # Main entry point
└── requirements.txt           # Dependencies
```

## Results and Analysis

The system provides three types of output:

1. **Model Performance**:

   - Training metrics and curves
   - Validation results
   - Cross-validation scores
   - Confusion matrices

2. **Review Analysis**:

   - Real-time predictions and confidence
   - Linguistic feature analysis
   - Pattern detection
   - Reasoning for predictions

3. **Summary Reports**:
   - Overall statistics
   - Performance metrics
   - Distribution analysis

## Dependencies

- TensorFlow 2.x
- Scikit-learn
- Kafka-Python
- NumPy
- Pandas
- PyYAML
- ApifyClient (for Amazon scraping)
- XGBoost
- NLTK

## Notes

- Enhanced feature extraction improves accuracy
- Batch normalization stabilizes training
- Class weights handle imbalanced data
- Model ensemble reduces prediction variance
- Cross-validation ensures robust evaluation
- Consistent feature names between training and inference prevents warnings

## Recent Improvements

- **Directory Structure**: Reorganized to separate Training, Testing, and Scraped data
- **Result Organization**: Created separate folders for training and testing results
- **Kafka Pipeline**: Added dedicated mode (`kafka-pipeline`) that runs scraper and consumer in one command
- **Feature Name Consistency**: Fixed warnings by ensuring feature names are consistent between training and inference
- **Real-time Results**: Centralized real-time predictions in the results directory for easier analysis

## Troubleshooting

### Docker Port Conflicts

If you encounter an error like this when running `docker-compose up -d`:

```
Error response from daemon: failed to set up container networking: driver failed programming external connectivity on endpoint [container_name] ([hash]): Bind for 0.0.0.0:32181 failed: port is already allocated
```

This means port 32181 (or another port) is already in use. You can resolve this by:

1. Stopping any services using that port:

   ```bash
   lsof -i :32181
   kill -9 [PID]
   ```

2. Or modifying the `docker-compose.yml` file to use different ports.

3. If you have existing orphaned containers, clean them up with:
   ```bash
   docker-compose up -d --remove-orphans
   ```
