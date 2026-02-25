#!/bin/bash
set -euo pipefail

MODE="development"
BUILD_MODE="reuse"
DETACH="false"

for arg in "$@"; do
    case "$arg" in
        prod|production)
            MODE="production"
            ;;
        --build)
            BUILD_MODE="build"
            ;;
        --rebuild|--no-cache)
            BUILD_MODE="rebuild"
            ;;
        --detach)
            DETACH="true"
            ;;
    esac
done

if [[ "$MODE" == "production" ]]; then
    echo "production"
    export BUILD_ENV=production
else
    echo "development"
    export BUILD_ENV=development
fi

UP_FLAGS=()
if [[ "$DETACH" == "true" ]]; then
    UP_FLAGS+=("-d")
fi

if [[ "$BUILD_MODE" == "rebuild" ]]; then
    echo "build mode: no-cache rebuild"
    docker compose build --no-cache
    docker compose up --force-recreate "${UP_FLAGS[@]}"
elif [[ "$BUILD_MODE" == "build" ]]; then
    echo "build mode: incremental build"
    docker compose up --build "${UP_FLAGS[@]}"
else
    echo "build mode: reuse existing image"
    docker compose up "${UP_FLAGS[@]}"
fi
