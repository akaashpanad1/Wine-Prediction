package org.njit.ap2835;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class Constants {
    private static final Logger logger = LogManager.getLogger(Constants.class);

    public static final String TRAINING_DATASET = "data/TrainingDataset.csv";
    public static final String VALIDATION_DATASET = "data/ValidationDataset.csv";
    public static final String MODEL_PATH = "data/TrainingModel";
    public static final String TESTING_DATASET = "data/TestDataset.csv";

    public static final String APP_NAME = "WineQualityTest";

    // Getter for the logger to make it accessible
    public static Logger getLogger() {
        return logger;
    }
}
