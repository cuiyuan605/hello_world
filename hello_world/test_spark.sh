#!/bin/bash
source /etc/profile
echo "test spark"

spark-submit --master yarn-cluster \
--conf spark.network.timeout=600 \
--conf spark.sql.shuffle.partitions=200 \
--conf spark.executor.memoryOverhead=2024 \
--conf spark.driver.memoryOverhead=2024 \
--executor-cores 4 \
--num-executors 2 \
--executor-memory 2g \
--driver-memory 1g \
test_spark.py