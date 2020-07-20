import numpy as np


class Firm(object):
    def __init__(self, prod_func: str = 'cobb-douglas', parameters: dict = {}):
        """ Class for the representative firm in the dynamic Solow model

        Parameters
        ----------
        prod_func   :   str, default='cobb-douglas'
            chosen production function. Options include:
                (1) The Cobb-Douglas production function. Parameters must
                include:
                    rho :   float - capital share of production

        parameters  :   dict, default={}
            keyword arguments for the production function (see prod_tech above)

        Attributes
        ----------
        production_history  :   dict
            production levels across time

        Public Methods
        --------------
        production
            returns the production and updates attributes based on the given
            inputs to production
        """
        self.production_history = []
        self.prod_func = prod_func
        self.param_kwargs = parameters

        prod_funcs = {
            'cobb-douglas': self._cobb_douglas,
            'leontief': self._leontief
        }

        try:
            self.production_function = prod_funcs[prod_func]
        except KeyError:
            'Production function not in available options'

    def production(self, inputs: dict):
        """ Update the production for the period

        Parameters
        ----------
        inputs  :   dict
            inputs to production depending on the function, see each production
            function's docstring

        Returns
        -------
        production  :   float
            current level of per capita production
        """
        curr_production = self.production_function(**inputs, **self.param_kwargs)
        self.production_history.append(curr_production)
        return curr_production

    def _cobb_douglas(self, k: float = 1, n: float = 1, tech: float = 1,
                      rho: float = 0.25):
        """ Cobb-Douglas production function Y=A K^rho N^(1-rho)

        Parameters
        ----------
        k       :   float, default = 1
            level of capital in production
        n       :   float, default = 1
            level of labour in production
        tech    :   float, default = 1
            level of technology for production
        rho     :   float, default = 0.25
            capital share in production

        Returns
        -------
        production  :   float
            per capita output to production
        """
        return tech * (k ** rho) * (n ** (1 - rho))

    def _leontief(self, k: float = 1, n: float = 1, tech: float = 1):
        """ Leontief production function Y=A max(K,N). In per capita case, n=1

        Parameters
        ----------
        k       :   float, default = 1
            level of capital in production
        n       :   float, default = 1
            level of labour in production
        tech    :   float, default = 1
            level of technology for production

        Returns
        -------
        production  :   float
            per capita output of production
        """
        return tech * np.maximum(k, n)
