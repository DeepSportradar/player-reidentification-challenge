from setuptools import setup, find_packages


setup(name='deepsportradar-player-reidentification',
      version='1.0.0',
      description='Deep Learning Library for Basketball Player Re-identification for the DeepSportRadar ReID Challenge', # FIXME
      author='Davide Zambrano & Vladimir Somers',
      author_email='d.zambrano@sportradar.com, v.somers@sportradar.com',
      url='https://github.com/DeepSportRadar/player-reidentification-challenge',
      license='MIT',
      install_requires=[
          'numpy', 'scipy', 'torch==1.8.1', 'torchvision',
          'six', 'h5py', 'Pillow',
          'scikit-learn', 'metric-learn'],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme'],
      },
      packages=find_packages(),
      keywords=[
          'Person Re-identification',
          'Computer Vision',
          'Deep Learning',
      ])
