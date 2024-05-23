#include "damotion/optimisation/program.h"

namespace damotion {
namespace optimisation {

const int &Program::NumberOfDecisionVariables() const {
  return x_manager_->NumberOfVariables();
}

void Program::AddDecisionVariable(const Variable &var) {
  x_manager_->AddVariable(var);
}

void Program::AddDecisionVariables(
    const Eigen::Ref<const VariableMatrix> &var) {
  x_manager_->AddVariables(var);
}

void Program::RemoveDecisionVariables(const Eigen::Ref<VariableMatrix> &var) {
  x_manager_->RemoveVariables(var);
}

bool Program::IsDecisionVariable(const Variable &var) {
  return x_manager_->IsVariable(var);
}

int Program::GetDecisionVariableIndex(const Variable &v) {
  return x_manager_->GetVariableIndex(v);
}

std::vector<int> Program::GetDecisionVariableIndices(const VariableVector &v) {
  return x_manager_->GetVariableIndices(v);
}

void Program::SetDecisionVariableVector() { x_manager_->SetVariableVector(); }

bool Program::SetDecisionVariableVector(const Eigen::Ref<VariableVector> &var) {
  return x_manager_->SetVariableVector(var);
}

bool Program::IsContinuousInDecisionVariableVector(const VariableVector &var) {
  return x_manager_->IsContinuousInVariableVector(var);
}

void Program::UpdateDecisionVariableBoundVectors() {
  x_manager_->UpdateVariableBoundVectors();
}

void Program::UpdateDecisionVariableInitialValueVector() {
  x_manager_->UpdateInitialValueVector();
}

const Eigen::VectorXd &Program::DecisionVariableInitialValues() const {
  return x_manager_->VariableInitialValues();
}

const Eigen::VectorXd &Program::DecisionVariableUpperBounds() const {
  return x_manager_->VariableUpperBounds();
}

const Eigen::VectorXd &Program::DecisionVariableLowerBounds() const {
  return x_manager_->VariableLowerBounds();
}

void Program::SetDecisionVariableBounds(const Variable &v, const double &lb,
                                        const double &ub) {
  x_manager_->SetVariableBounds(v, lb, ub);
}
void Program::SetDecisionVariableBounds(const VariableVector &v,
                                        const Eigen::VectorXd &lb,
                                        const Eigen::VectorXd &ub) {
  x_manager_->SetVariableBounds(v, lb, ub);
}

void Program::SetDecisionVariableInitialValue(const Variable &v,
                                              const double &x0) {
  x_manager_->SetVariableInitialValue(v, x0);
}
void Program::SetDecisionVariableInitialValue(const VariableVector &v,
                                              const Eigen::VectorXd &x0) {
  x_manager_->SetVariableInitialValue(v, x0);
}

void Program::ListDecisionVariables() { x_manager_->ListVariables(); }

const int &Program::NumberOfParameters() const {
  return p_manager_->NumberOfVariables();
}

void Program::AddParameter(const Variable &var) {
  p_manager_->AddVariable(var);
}

void Program::AddParameters(const Eigen::Ref<const VariableMatrix> &var) {
  p_manager_->AddVariables(var);
}

void Program::RemoveParameters(const Eigen::Ref<VariableMatrix> &var) {
  p_manager_->RemoveVariables(var);
}

bool Program::IsParameter(const Variable &var) {
  return p_manager_->IsVariable(var);
}

int Program::GetParameterIndex(const Variable &v) {
  return p_manager_->GetVariableIndex(v);
}

std::vector<int> Program::GetParameterIndices(const VariableVector &v) {
  return p_manager_->GetVariableIndices(v);
}

void Program::SetParameterVector() { p_manager_->SetVariableVector(); }

bool Program::SetParameterVector(const Eigen::Ref<VariableVector> &var) {
  return p_manager_->SetVariableVector(var);
}

bool Program::IsContinuousInParameterVector(const VariableVector &var) {
  return p_manager_->IsContinuousInVariableVector(var);
}

void Program::UpdateParameterValueVector() {
  p_manager_->UpdateInitialValueVector();
}

const Eigen::VectorXd &Program::ParameterValues() const {
  return p_manager_->VariableInitialValues();
}

void Program::SetParameterValue(const Variable &p, const double &p0) {
  p_manager_->SetVariableInitialValue(p, p0);
}
void Program::SetParameterValue(const VariableVector &p,
                                const Eigen::VectorXd &p0) {
  p_manager_->SetVariableInitialValue(p, p0);
}

void Program::ListParameters() { p_manager_->ListVariables(); }

}  // namespace optimisation
}  // namespace damotion
