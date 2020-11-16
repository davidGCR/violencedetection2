
# sys.path.insert(1,'/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2')
# root = '/media/david/datos/Violence DATA/'
# root = '/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2'

import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(dirname)
# sys.path.append(dirname)
# subprocess.run(['python3', 'import sys', 'IN_COLAB = 'google.colab' in sys.modules'])

def getRoot():
  # IN_COLAB = 'google.colab' in sys.modules
  # print(IN_COLAB)
  if dirname == '/content':
      root = '/content'
      enviroment = 'colab'
  else:
      root = '/Users/davidchoqueluqueroman/Desktop/PROJECTS-SOURCE-CODES/violencedetection2'
      enviroment = 'local'
  return root, enviroment

root, enviroment = getRoot()
# print('root:', root)

# root = '/content/'
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/violencedetection')
