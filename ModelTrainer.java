package org.njit.ap2835;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.IOException;

import static org.njit.ap2835.Constants.*;

public class ModelTrainer {

    public static void main(String[] args) {
        configureLoggerLevels();

        SparkSession spark = createSparkSession();

        File tempFile = new File(TRAINING_DATASET);
        boolean exists = tempFile.exists();
        if (exists) {
            ModelTrainer trainer = new ModelTrainer();
            trainer.logisticRegression(spark);
        } else {
            System.out.print("TrainingDataset.csv doesn't exist");
        }
    }

    private static void configureLoggerLevels() {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
        Logger.getLogger("com.amazonaws.auth").setLevel(Level.DEBUG);
        Logger.getLogger("com.github").setLevel(Level.ERROR);
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder()
                .appName(APP_NAME)
                .master("local[*]")
                .config("spark.executor.memory", "2147480000")
                .config("spark.driver.memory", "2147480000")
                .config("spark.testing.memory", "2147480000")
                .getOrCreate();
    }

    public void logisticRegression(SparkSession spark) {
        System.out.println();
        Dataset<Row> labeledFeatureDf = getDataFrame(spark, true, TRAINING_DATASET).cache();
        LogisticRegression logReg = new LogisticRegression().setMaxIter(100).setRegParam(0.0);

        Pipeline pl1 = new Pipeline();
        pl1.setStages(new PipelineStage[]{logReg});

        PipelineModel model1 = pl1.fit(labeledFeatureDf);

        LogisticRegressionModel lrModel = (LogisticRegressionModel) (model1.stages()[0]);
        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
        double accuracy = trainingSummary.accuracy();
        double fMeasure = trainingSummary.weightedFMeasure();

        System.out.println();
        System.out.println("Training DataSet Metrics ");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("F-measure: " + fMeasure);

        Dataset<Row> testingDf1 = getDataFrame(spark, true, VALIDATION_DATASET).cache();
        Dataset<Row> results = model1.transform(testingDf1);

        System.out.println("\nValidation Training Set Metrics");
        results.select("features", "label", "prediction").show(5, false);
        printMetrics(results);

        try {
            model1.write().overwrite().save(MODEL_PATH);
        } catch (IOException e) {
            logger.error(e);
        }
    }

    public void printMetrics(Dataset<Row> predictions) {
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1: " + f1);
    }

    public Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String name) {

        Dataset<Row> validationDf = spark.read().format("csv")
                .option("header", "true")
                .option("multiline", true)
                .option("sep", ";")
                .option("quote", "\"")
                .option("dateFormat", "M/d/y")
                .option("inferSchema", true)
                .load(name);

        validationDf = renameColumns(validationDf).cache();

        Dataset<Row> labeledFeatureDf = validationDf
                .select("label", "alcohol", "sulphates", "pH", "density", "free_sulfur_dioxide",
                        "total_sulfur_dioxide", "chlorides", "residual_sugar", "citric_acid",
                        "volatile_acidity", "fixed_acidity")
                .na().drop().cache();

        VectorAssembler assembler = createFeatureAssembler();
        if (transform) {
            labeledFeatureDf = assembler.transform(labeledFeatureDf).select("label", "features");
        }

        return labeledFeatureDf;
    }

    private Dataset<Row> renameColumns(Dataset<Row> validationDf) {
        String[] columnNames = {
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates",
                "alcohol", "quality"
        };

        for (int i = 0; i < columnNames.length; i++) {
            validationDf = validationDf.withColumnRenamed(columnNames[i], StringUtils.remove(columnNames[i], ' '));
        }

        return validationDf;
    }

    private VectorAssembler createFeatureAssembler() {
        return new VectorAssembler().setInputCols(new String[]{"alcohol", "sulphates", "pH", "density",
                "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
                "citric_acid", "volatile_acidity", "fixed_acidity"}).setOutputCol("features");
    }
}
