#!/bin/env python

# Import Dependencies
from datetime import datetime, timezone
from kafka import KafkaConsumer, KafkaProducer
import logging
import os
import json

# Mothra dependencies
from mothra.misc import AlbumentationsTransform, label_func, _generate_parser
from skimage.io import imread


def timestamp_now():
    timestamp = str(datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"))
    timestamp_cleaned = timestamp[:-3]
    timestamp_timezone = timestamp_cleaned + 'Z'
    return timestamp_timezone


def send_updated_opends(annotations: list, producer: KafkaProducer) -> None:
    for annotation in annotations:
        logging.info('Publishing annotation: ' + str(annotation))
        producer.send('annotation', annotation)


def create_annotation(data_elements: dict, json_value: dict):
    logging.info('Successfully completed Mothra operation, preparing annotations...')

    annotations = []
    for key, value in data_elements.items():
        if json_value.get('data').get(key) is None:
            annotations.append(
                {
                    'type': 'Annotation',
                    'motivation': 'https://hdl.handle.net/adding',
                    'creator': 'https://hdl.handle.net/enrichment-service-pid',
                    'created': timestamp_now(),
                    'target': {
                        'id': json_value.get('id'),
                        'type': 'https://hdl.handle.net/21...',
                        'indvProp': key
                    },
                    'body': {
                        'source': json_value.get('data').get('ac:accessURI'),
                        'values': value,
                    }
                })
    return annotations


def mothra(image_uri):
    """
    Main function for running the Mothra services.
    Images are received via the image URL.
    Returns additional info object of the specimen of the provided image
    """

    args = _generate_parser()

    stages = ['ruler_detection', 'binarization', 'measurements']

    if args.stage not in stages:
        logging.info(f"* mothra expects stage to be 'ruler_detection', "
              f"binarization', or 'measurements'. Received '{args.stage}'")
        return None

    # Set up caching and import Mothra modules
    if args.cache:
        from mothra import cache
        import joblib
        cache.memory = joblib.Memory('./cachedir', verbose=0)

    from mothra import (ruler_detection, tracing, measurement, binarization,
                        identification, misc, preprocessing)

    stage_idx = stages.index(args.stage)
    pipeline_process = stages[:stage_idx + 1]

    # reading and processing input path.
    input_name = args.input
    image_paths = misc.process_paths_in_input(input_name)

    number_of_images = len(image_paths)

    additional_info = {}

    try:
        logging.info(f'\nImage {1}/{number_of_images} : {image_uri}')

        image_rgb = imread(image_uri)

        # check image orientation and untilt it, if necessary.
        if args.auto_rotate:
            image_rgb = preprocessing.auto_rotate(image_rgb, image_uri)

        for step in pipeline_process:
            # first, binarize the input image and return its components.
            _, ruler_bin, lepidop_bin = binarization.main(image_rgb)

            if step == 'ruler_detection':
                t_space, top_ruler = ruler_detection.main(image_rgb, ruler_bin)

            elif step == 'measurements':
                points_interest = tracing.main(lepidop_bin)
                _, dist_mm = measurement.main(points_interest, t_space)
                # measuring position and gender
                position, gender, probabilities = identification.main(image_rgb)

                additional_info = dist_mm

                if position:
                    additional_info['position'] = position
                if gender:
                    additional_info['gender'] = gender
                if probabilities:
                    additional_info['probabilities'] = probabilities
    except Exception as exc:
        logging.error(f"* Sorry, could not process {image_uri}. More details:\n {exc}")

    return additional_info


def start_kafka() -> None:
    """
    Start a kafka listener and process the messages by unpacking the image.
    When done it will republish the object, so it can be validated and storage by the processing service
    """
    consumer = KafkaConsumer(os.environ.get('KAFKA_CONSUMER_TOPIC'), group_id=os.environ.get('KAFKA_CONSUMER_GROUP'),
                             bootstrap_servers=[os.environ.get('KAFKA_CONSUMER_HOST')],
                             value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                             enable_auto_commit=True,
                             max_poll_interval_ms=50000,
                             max_poll_records=10)
    producer = KafkaProducer(bootstrap_servers=[os.environ.get('KAFKA_PRODUCER_HOST')],
                             value_serializer=lambda m: json.dumps(m).encode('utf-8'))

    for msg in consumer:
        try:
            json_value = msg.value
            object_id = json_value.get('id')
            logging.info(f'Received message for id: {object_id}')
            image_uri = json_value['data']['ac:accessURI']
            data_elements = mothra(image_uri)
            annotations = create_annotation(data_elements, json_value)
            logging.info(f'Publishing the result: {object_id}')
            send_updated_opends(annotations, producer)
        except:
            logging.exception(f'Failed to process message: {msg}')


if __name__ == "__main__":
    start_kafka()
