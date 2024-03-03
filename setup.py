from distutils.core import setup
setup \
  ( name='cpcarl'
  , version='0.1'
  , packages=['cpcarl']
  , install_requires= \
    [ "torch"
    , "numpy"
    , "matplotlib"
    , "cpplot @ git+https://github.com/cspollard/cpplot@bd79f7986cf901dc5f3eab32a31e4e033d853343"
    ]
  )