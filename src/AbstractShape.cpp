#include "AbstractShape.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace composite
        {

            std::unordered_map<std::string, int> interKeyDict;

            AbstractShape::AbstractShape()
            {
                type = "Shape";
                if (interKeyDict.empty())
                {
                    interKeyDict.insert(std::make_pair<std::string, int>("Box", 0));
                    interKeyDict.insert(std::make_pair<std::string, int>("Cylinder", 1));
                    interKeyDict.insert(std::make_pair<std::string, int>("Pyramid", 2));
                    interKeyDict.insert(std::make_pair<std::string, int>("Sphere", 3));
                }
            }

        }

    }

}
