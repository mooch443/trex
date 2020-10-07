git submodule init
git submodule update
git -C Application/src/commons submodule init
git -C Application/src/commons submodule update

sudo apt install build-essential libudev-dev mesa-common-dev freeglut3 freeglut3-dev libopenal-dev libjpeg-dev libvorbis-dev libflac-dev
sudo apt install autoconf libtool texinfo libxrandr-dev libfreetype6-dev gcc-8 g++-8 libzip-dev python3-dev

GRAY='\033[0;37m'
RED='\033[0;31m'
GREEN='\033[1;32m'
WHITE='\033[1;37m'
NC='\033[0m'

function install_all_alternatives {
    versions=$(ls /usr/bin/ | grep ^gcc-[0-9] | cut -c5-)
    for f in $versions
    do 
        echo -e "Installing ${GRAY}GCC $f${NC} alternative..."
        if ! sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$f 2 --slave /usr/bin/g++ g++ /usr/bin/g++-$f
        then
            return 1
        fi
    done
    
    return 0
}