#!/bin/bash

# Takes three arguments:
#   bucket name - one that has already been created
#   name of key file - without .pem extension
#   number of slave instances
#      ex. bash launch_cluster.sh mybucket mypem 2

# Requires the awscli to be set up, need to have correct default region configured

# aws s3 cp bootstrap.sh s3://$1/bootstrap.sh

aws emr create-cluster \
    --name FinalProjecTaxiTestCluster \
    --release-label emr-5.2.0 \
    --applications Name=Hadoop \
        Name=Hive \
        Name=Pig \
        Name=Hue \
        Name=Spark \
        Name=Ganglia \
        Name=Zeppelin \
    --ec2-attributes KeyName=$2 \
    --use-default-roles \
    --instance-groups \
      InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m3.xlarge \
      InstanceGroupType=CORE,InstanceCount=$3,InstanceType=r3.2xlarge \
    --bootstrap-actions Path=s3://$1/bootstrap.sh


