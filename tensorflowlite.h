#ifndef TENSORFLOW_H
#define TENSORFLOW_H

#include <QStringList>
#include <QImage>
#include <QRectF>

#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/graph_info.h"
#include "tensorflow/contrib/lite/kernels/register.h"

using namespace tflite;

class TensorflowLite
{
public:
    static const int knIMAGE_CLASSIFIER = 1;
    static const int knOBJECT_DETECTION = 2;

    TensorflowLite();

    bool init();
    double getThreshold() const;
    void setThreshold(double value);
    QStringList getResults();
    QList<double> getConfidence();
    QList<QRectF> getBoxes();
    int getKindNetwork();
    bool run(QImage img);
    QString getModelFilename() const;
    void setModelFilename(const QString &value);
    QString getLabelsFilename() const;
    void setLabelsFilename(const QString &value);
    int getImgHeight() const;
    int getImgWidth() const;
    double getInfTime() const;

private:
    // Box prior
    static const int NUM_LINES   = 4;
    static const int NUM_RESULTS = 1917; // WARNING: can this value be obtained from the model?
    // Fixed image size for image classification
    const int fixed_width  = 224;
    const int fixed_heigth = 224;
    // Output names
    const QString num_detections    = "num_detections";
    const QString detection_classes = "detection_classes";
    const QString detection_scores  = "detection_scores";
    const QString detection_boxes   = "detection_boxes";
    // Output lists
    const std::vector<std::string> listOutputsObjDet = {num_detections.toStdString(),detection_classes.toStdString(),detection_scores.toStdString(),detection_boxes.toStdString()};
    const std::vector<std::string> listOutputsImgCla = {"MobilenetV2/Predictions/Reshape_1"};

    bool initialized;
    double threshold;

    // Results
    QStringList rCaption;
    QList<double> rConfidence;
    QList<QRectF> rBox;
    double infTime;

    int kind_network;
    std::vector<TfLiteTensor *> outputs;
    std::unique_ptr<Interpreter> interpreter;
    std::unique_ptr<FlatBufferModel> model;
    ops::builtin::BuiltinOpResolver resolver;
    StderrReporter error_reporter;

    int wanted_height, wanted_width, wanted_channels;

    bool inference();
    bool setInputs(QImage image);
    bool getClassfierOutputs(std::vector<std::pair<float, int> > *top_results);
    bool getObjectOutputs(QStringList &captions, QList<double> &confidences, QList<QRectF> &locations);
    bool readLabels();

    QString input_name;
    TfLiteType input_dtype;
    std::unique_ptr<TfLiteTensor> input_tensor;
    QString modelFilename;
    QString labelsFilename;
    QStringList labels;
    QString getLabel(int index);
    int img_height, img_width, img_channels;
    const QImage::Format format = QImage::Format_RGB888;
    const int numChannels = 3;
};

#endif // TENSORFLOW_H
