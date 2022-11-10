{
    pkgs ? import <nixpkgs> {},
    stdenv ? pkgs.stdenv
}:

let 
  #opencvGtk = opencv.override (old : { enableGtk2 = true; });
  opencvGtk = pkgs.opencv;
in 
  stdenv.mkDerivation {
    name = "nix-inference-test";
    src = ./.;
  
    nativeBuildInputs = with pkgs; [ cmake ];
    buildInputs = with pkgs; [ opencvGtk onnxruntime ];
  
    buildPhase =
    ''
      rm -rf build
      cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES .
      make
    '';
  
    installPhase = ''
      echo out is $out
      echo pwd is $(pwd)
      mkdir -p $out/bin
      cp onnx_test $out/bin/
    '';
  
    # shellHook = ''
    #   export CMAKE_C_COMPILER=${pkgs.clang}/bin/clang
    #   export CMAKE_CXX_COMPILER=${pkgs.clang}/bin/clang++
    #   export PATH_CLANGD=${clang}/bin/clangd
    # '';
  }
