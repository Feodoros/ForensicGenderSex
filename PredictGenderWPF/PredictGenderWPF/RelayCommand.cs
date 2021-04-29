using System;
using System.Windows.Input;

namespace PredictGenderWPF
{
    public class RelayCommand : ICommand
    {

        Action<object> _executemethod;
        Func<object, bool> _canexecutemethod;

        public RelayCommand(Action<object> executemethod, Func<object, bool> canexecutemethod)
        {
            _executemethod = executemethod;
            _canexecutemethod = canexecutemethod;
        }


        public bool CanExecute(object parameter)
        {
            if (_canexecutemethod != null)
            {
                return _canexecutemethod(parameter);
            }
            else
            {
                return false;
            }
        }

        public event EventHandler CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }

        public void Execute(object parameter)
        {
            _executemethod(parameter);
        }
    }
}
