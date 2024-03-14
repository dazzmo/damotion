#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "common/types.h"

class Controller {
   public:
    VectorXd u() { return u_; }

   protected:
    VectorXd u_;

   private:
};

#endif /* CONTROLLER_H */
