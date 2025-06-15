#!/bin/sh

if [ ! -d "/TopSpotifAI/models/RidgeRegressor/out" ]; then
    python /TopSpotifAI/models/RidgeRegressor/test.py
fi

if [ ! -d "/TopSpotifAI/models/EnhancedMLP/out" ]; then
    python /TopSpotifAI/models/EnhancedMLP/test.py
fi
echo "Models Trained!"
tail -f /dev/null