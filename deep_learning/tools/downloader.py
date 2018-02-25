import os
import sys
import tarfile

from six.moves import urllib

DEST_DIRECTORY = '/tmp'
# pylint: disable = line-too-long
SOURCE_DATA = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable = line-too-long

def download():
  if not os.path.exists(DEST_RIRECTORY):
    os.makedirs(DEST_DIRECTORY)
    file_name = SOURCE_DATA.split('/')[-1]
    file_path = os.path.join(DEST_DIRECTORY, file_name)

  if not os.path.exists(file_path):
    def _progress(count, block_size, total_size):
      sys.stdout.write(
        '\r>> Downloading %s %.1f%%' % (
          file_name,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    file_path, _ = urllib.request.urlretrieve(DATA_URL, file_path, _progress)
    print()
    statinfo = os.stat(file_path)
    print(
      'Successfully downloaded',
      file_name,
      statinfo.st_size,
      'bytes.')
  tarfile.open(file_path, 'r:gz').extractall(DEST_DIRECTORY)
