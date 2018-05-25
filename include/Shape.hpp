class Shape
{
    public:
        virtual ~Shape() { ; }
        virtual Shape* structFactory() = 0;
}
