import { Component } from "react";
import {v4 as uuidv4} from 'uuid';

export default class Form extends Component{

    render(){
        return (
            <form>
                <input type = 'text' placeholder = 'Enter a Todo..' className='task-input'
                />
                <button className='button-add' type = 'submit'>Add</button>
            </form>

        )
        }

} // Component parenthesis 
