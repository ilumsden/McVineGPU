#ifndef SYSVARS_HPP
#define SYSVARS_HPP

/* This file stores the extern declarations of any system-wide
 * global variables that might be needed.
 */

namespace mcvine
{

    namespace gpu
    {

        // The attenuation coefficient for the scattering material
        // Unit: meters
        extern float atten;

        // The mass of a neutron
        // Unit: kg
        const float m_neutron = 1.6749286e-27;

    }

}

#endif
