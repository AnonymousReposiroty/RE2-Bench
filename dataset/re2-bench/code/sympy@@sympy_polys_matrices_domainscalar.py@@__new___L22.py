from sympy.polys.domains import Domain, ZZ

class DomainScalar:
    """
    docstring
    """

    def __new__(cls, element, domain):
        if not isinstance(domain, Domain):
            raise TypeError('domain should be of type Domain')
        if not domain.of_type(element):
            raise TypeError('element %s should be in domain %s' % (element, domain))
        return cls.new(element, domain)

    @classmethod
    def new(cls, element, domain):
        obj = super().__new__(cls)
        obj.element = element
        obj.domain = domain
        return obj
