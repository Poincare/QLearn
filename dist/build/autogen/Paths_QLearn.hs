module Paths_QLearn (
    version,
    getBinDir, getLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []
bindir, libdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/Users/dhaivat/.cabal/bin"
libdir     = "/Users/dhaivat/.cabal/lib/x86_64-osx-ghc-7.10.3/QLearn-0.1.0.0-LcrniGKynVQAV06Td4T6Uo"
datadir    = "/Users/dhaivat/.cabal/share/x86_64-osx-ghc-7.10.3/QLearn-0.1.0.0"
libexecdir = "/Users/dhaivat/.cabal/libexec"
sysconfdir = "/Users/dhaivat/.cabal/etc"

getBinDir, getLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "QLearn_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "QLearn_libdir") (\_ -> return libdir)
getDataDir = catchIO (getEnv "QLearn_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "QLearn_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "QLearn_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
