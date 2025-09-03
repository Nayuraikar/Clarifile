#!/bin/bash
BASE="http://127.0.0.1:4000"

echo "SCAN..."
curl -s -X POST "$BASE/scan" | jq .

echo -e "\nEMBED..."
curl -s -X POST "$BASE/embed" | jq .

echo -e "\nREINDEX..."
curl -s -X POST "$BASE/reindex" | jq .

echo -e "\nPROPOSALS..."
curl -s -X GET "$BASE/proposals" | jq .

echo -e "\nDone."
