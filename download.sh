#!/bin/bash

# segmentation
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1-e6Kuoj3qv12MELyYIHZQNsLnKQ3EzHO" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1-e6Kuoj3qv12MELyYIHZQNsLnKQ3EzHO" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

# depth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1J4FANOXevJd-egbhGVeK9gzFQtqJamw4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1J4FANOXevJd-egbhGVeK9gzFQtqJamw4" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

# depth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1wTkTB81_SkTF2R00i-qxtPGR_b365l8S" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1wTkTB81_SkTF2R00i-qxtPGR_b365l8S" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
