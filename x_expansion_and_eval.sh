OBJECT_DETECTION_DIR=/data/venv-tensorflow2/tensorflow-models/research/object_detection/
cd $OBJECT_DETECTION_DIR

OID_DIR=/data/venv-tensorflow2/open-images-dataset/
HIERARCHY_FILE=${OID_DIR}/annotation-instance-segmentation/metadata/challenge-2019-label300-segmentable-hierarchy.json
BOUNDING_BOXES=${OID_DIR}/annotation-instance-segmentation/validation/challenge-2019-validation-segmentation-bbox
IMAGE_LABELS=${OID_DIR}/annotation-instance-segmentation/validation/challenge-2019-validation-segmentation-labels
INSTANCE_SEGMENTATIONS=${OID_DIR}/annotation-instance-segmentation/validation/challenge-2019-validation-masks/challenge-2019-validation-segmentation-masks

python dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${BOUNDING_BOXES}.csv \
    --output_annotations=${BOUNDING_BOXES}_expanded.csv \
    --annotation_type=1

python dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${IMAGE_LABELS}.csv \
    --output_annotations=${IMAGE_LABELS}_expanded.csv \
    --annotation_type=2

python dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${INSTANCE_SEGMENTATIONS}.csv \
    --output_annotations=${INSTANCE_SEGMENTATIONS}_expanded.csv \
    --annotation_type=1

#INPUT_PREDICTIONS=/path/to/instance_segmentation_predictions.csv
#OUTPUT_METRICS=/path/to/output/metrics/file
#
#python ${OBJECT_DETECTION_DIR}/metrics/oid_challenge_evaluation.py \
#    --input_annotations_boxes=${BOUNDING_BOXES}_expanded.csv \
#    --input_annotations_labels=${IMAGE_LABELS}_expanded.csv \
#    --input_class_labelmap=object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt \
#    --input_predictions=${INPUT_PREDICTIONS} \
#    --input_annotations_segm=${INSTANCE_SEGMENTATIONS}_expanded.csv
#    --output_metrics=${OUTPUT_METRICS} \