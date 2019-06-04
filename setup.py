from distutils.core import setup  # noqa

setup(
    name='capsnet',
    description='CapsNet - face recognition',
    version='1.0.1',
    author='Tomas Coufal',
    packages=['capsnet',],
    license='Apache 2.0',
    url='https://github.com/tumido/capsnet-face',
    install_requires=['tensorflow', 'scikit-learn', 'keras', 'numpy', 'Pillow', 'click'],
    entry_points='''
        [console_scripts]
        capsnet=capsnet.cli:cli
    ''',
)
