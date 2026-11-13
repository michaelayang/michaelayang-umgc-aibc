#!/bin/bash -v

minikube start --driver=docker
eval $(minikube docker-env)
minikube status
kubectl get node
minikube image build -t local-myang-mathbot-python:v1 .
kubectl apply -f k8s-manifest.yaml
kubectl get all
kubectl get pod
minikube ip
kubectl get node -o wide
minikube dashboard
minikube tunnel
