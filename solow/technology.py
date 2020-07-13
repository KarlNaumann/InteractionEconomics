import numpy as np

class TechProcess(object):
    def __init__(self, process: str = 'deterministic', parameters: dict = {},
                 start: float = 1):
        """ Class that defines the technological process method. This can

        Parameters
        ----------
        process :   str, default 'deterministic'
            Define which kind of technological process is desired. The current
            options are:
                - deterministic - growth at rate epsilon, p_kwds must include
                'epsilon'
        parameters :    dict, default={}
            Keyword parameter dict for the chosen process (see process)
        start   :   float, default=1
            starting value for the level of technology

        Attributes
        ----------
        process_type    :   str
            selected technology process
        technology  :   float
            current level of technology
        levels  :   list
            technology levels across periods

        Methods
        -------
        update
            update the technology level from one period to the next

        get_levels
            return the technology levels as an np.array
        """

        options = {
            'deterministic': self.deterministic
        }

        self.process_type = process
        if isinstance(parameters, dict):
            self.parameters = parameters
        else:
            self.parameters = {}

        try:
            self.process_update = options[process]
        except KeyError:
            "Selected process type not in the options"

        self.technology = start
        self.levels = [start]

    def update(self):
        """Simple function as placeholder to update the chosen process"""
        self.process_update(**self.parameters)

    def full_tech(self, periods: int, start: float):
        """Generate the full array of technological progress and return it

        Parameters
        ----------
        periods :   int
            number of periods to iterate along
        start   :   float
            starting  value for the technology process

        Returns
        -------
        tech_process    :   np.array
            array of floats describing the development of technology
        """
        # Save current state to reinstate later
        save_level = self.technology
        save_levels = self.levels

        # Develop tech progress
        self.technology = start
        self.levels = [start]
        for i in range(periods):
            self.update()

        tech_process = np.array(self.levels)
        self.technology = save_level
        self.levels = save_levels

        return tech_process

    def deterministic(self, epsilon: float, a0: float):
        """ Update the deterministic trend

        Parameters
        ----------
        epsilon :   float
            exogenous technology growth rate

        Returns
        -------
        level   :   float
            level of technology after the update
        """
        self.technology *= np.exp(epsilon)
        self.levels.append(self.technology)
