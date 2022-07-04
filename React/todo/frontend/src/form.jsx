import { Component } from "react";

export default class Form extends Component{
    
    constructor(props){
        super(props);
        this.state = {
            task : '',
        }
    }

    render(){
        return (
            <form>
                <input type = 'text' placeholder = 'Enter a Todo..' className='task-input' value = {this.state.task} onChange = {this.handletaskchange.bind(this)}/>
                <button className='button-add' type = 'submit' onClick ={ () => {
                        this.handlebuttonclick.bind(this)}}>Add</button>
            </form>

        )
        }

handletaskchange = (event) => {

    var currtask = this.state.task

    // Extract the value of the input element represented by `target`
    var modifiedValue = event.target.value
  
    currtask = modifiedValue;
  
    // Update the state object
    this.setState({
      task: currtask
    })
  }


handlebuttonclick = () => {

    console.log(this.state.task)
    return (
    <li className = 'list-item'>
        <a value={this.state.task} className= 'list'>
        </a>
    </li>
    )

}


} // Component parenthesis 
